"""Select and reconstruct natural contiguous occlusion intervals, without GT.

A's whole-video page model only qualifies stable time intervals. The interval's
A selection, terminal ink/absence evidence, and B RGB donors are then rebuilt
from interval observations alone. A clear outside observation is evaluation only.
"""
import argparse,sys,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.pages import observations,group_pages
from lecture2doc.page_restore import restore_page
from lecture2doc.vision import Config,small,measure_features
from lecture2doc.storage import read,write,digest
from accessmath_run import hardlink_or_copy
from accessmath_evaluate import predicted_ink,score
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-cases',type=int,default=4);a=p.parse_args()
source=a.record.resolve();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);m=read(source/'manifest.json');config=Config(**m['config']);config.reconstruction=True
shutil.copytree(source/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
frames=m['source_frames'];by_id={f['id']:f for f in frames};indices={f['id']:i for i,f in enumerate(frames)}
old_cache=next((source/'cache/pages').iterdir())
data=observations(source,old_cache,frames,config);shape=tuple(map(int,data['shape']));size=int(np.prod(shape))
def unpack(key,i):return np.unpackbits(data[key][i],count=size).reshape(shape).astype(bool)
proposals=[]
for page in m['board_pages']:
 target=cv2.imread(str(source/page['terminal_ink_mask']),0)>0
 board=cv2.imread(str(source/page['analysis_board_mask']),0)>0
 core=cv2.erode(board.astype('uint8'),np.ones((25,25),np.uint8)).astype(bool);target &= core
 if target.sum()<200:continue
 ids=[fid for fid in page['source_frame_ids'] if by_id[fid]['time']>=page['end']-120]
 runs=[];run=[]
 for fid in ids:
  i=indices[fid];fg=cv2.dilate(unpack('mask',i).astype('uint8'),np.ones((7,7),np.uint8)).astype(bool)
  ink=unpack('ink',i);near=cv2.dilate(ink.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
  missing=(target &~fg &~near).sum()/max(1,(target&~fg).sum())
  hidden=int((target &fg).sum())
  eligible=missing<=.01 and hidden>=max(20,int(target.sum()*.02))
  if eligible:run.append((fid,fg,hidden))
  else:
   if len(run)>=8:runs.append(run)
   run=[]
 if len(run)>=8:runs.append(run)
 for run in runs:
  # Bounded continuous windows, never cherry-picked non-contiguous donor frames.
  for begin in range(0,len(run),24):
   window=run[begin:begin+40]
   if len(window)<8:continue
   anchor=min(window,key=lambda f:f[2]);visible_votes=sum((~fg).astype('uint16') for _,fg,_ in window)
   potential=int((target & anchor[1] & (visible_votes>=2)).sum())
   if potential<12:continue
   proposals.append(dict(parent_page_id=page['id'],frame_ids=[f[0] for f in window],start=by_id[window[0][0]]['time'],end=by_id[window[-1][0]]['time']+config.sample_seconds,
     qualification_hidden_pixels=anchor[2],qualification_recoverable_pixels=potential,score=potential,reference_frame_id=page['selected_frame_id'],parent_target=page['terminal_ink_mask'],parent_board=page['analysis_board_mask']))
proposals.sort(key=lambda r:r['score'],reverse=True)
# Diverse pages first; each selection is frozen before calling B or scoring it.
chosen=[];used=set()
for r in proposals:
 if r['parent_page_id'] in used:continue
 chosen.append(r);used.add(r['parent_page_id'])
 if len(chosen)>=a.max_cases:break
if len(chosen)<a.max_cases:
 for r in proposals:
  if r in chosen or any(not(r['end']<=c['start'] or r['start']>=c['end']) for c in chosen):continue
  chosen.append(r)
  if len(chosen)>=a.max_cases:break
write(root/'frozen-inputs.json',dict(ground_truth_used=False,record=str(source),policy='Continuous >=4s intervals, every sampled frame occludes >=2% of central terminal writing; stable co-visible content; at least 12 recoverable ink pixels. Ranked before restoration.',proposals=chosen))
results=[]
for number,proposal in enumerate(chosen):
 folder=root/f'case-{number:02d}';folder.mkdir(exist_ok=True)
 selected_frames=[by_id[fid] for fid in proposal['frame_ids']];inds=[indices[f['id']] for f in selected_frames]
 subset=dict(ink=data['ink'][inds],mask=data['mask'][inds],sharp=data['sharp'][inds],shape=shape)
 # Rebuild the interval evidence without the clear reference or whole-page ink.
 observed=group_pages(selected_frames,subset,config,proposal['end'],(),folder)
 if len(observed['pages'])!=1:
  results.append(dict(proposal=proposal,status='interval_not_a_single_state',pages=len(observed['pages'])));continue
 page=observed['pages'][0]
 for field,suffix in [('terminal_ink_mask','.target.png'),('analysis_board_mask','.board.png'),('terminal_absence_evidence','.evidence.npz')]:page[field]=(folder/(page['id']+suffix)).relative_to(root).as_posix()
 state=dict(id='interval',scene=0,start=proposal['start'],end=proposal['end'],source_frame_ids=proposal['frame_ids'],quality_flags=[],boundary_reason='end_of_interval')
 r=restore_page(root,folder,page,by_id,config,observed['policy'],[state])
 # Held-out RGB is read only after the restoration has returned.
 reference=cv2.imread(str(source/by_id[proposal['reference_frame_id']]['image']))
 A=cv2.imread(str(root/by_id[page['selected_frame_id']]['image']));B=cv2.imread(str(root/r['image']))
 board=cv2.imread(str(source/proposal['parent_board']),0)>0
 H,W=reference.shape[:2];valid=cv2.resize(board.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 truth=predicted_ink(reference,(H,W),10)&valid
 metrics={name:score(predicted_ink(im,(H,W),10),truth,valid) for name,im in [('A',A),('B',B)]}
 row=dict(proposal=proposal,page=page,result=r,metrics=metrics,status=r['status'])
 images=[A,B,reference];labels=[f"Actual interval A {proposal['start']:.1f}-{proposal['end']:.1f}s",'B: '+r['status'],'Outside reference (not official GT)']
 sheet=Image.new('RGB',(1440,300),'white');draw=ImageDraw.Draw(sheet)
 for j,(im,label) in enumerate(zip(images,labels)):
  sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,28));draw.text((j*480+4,6),label,fill='black')
 sheet.save(folder/'comparison.jpg',quality=95);row['comparison']=(folder/'comparison.jpg').relative_to(root).as_posix();results.append(row)
 write(root/'results.json',dict(ground_truth_used_for_selection=False,clear_reference_withheld_from_reconstruction=True,record=str(source),results=results))
 print(number,proposal['start'],proposal['end'],r['status'],r['reason'],metrics,flush=True)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>실제 연속 가림 구간 · 구간 관측만으로 A/B</h1><p>공식 GT가 아닌 보류한 원본 참고 영상과 비교합니다. 정상 전체영상 평가와 별도 실험입니다.</p>'+''.join(f'<h2>{r["status"]}</h2><img style="max-width:100%" src="{r.get("comparison","")}"><pre>{r.get("metrics",{})}</pre>' for r in results))
print('cases',len(results),flush=True)
