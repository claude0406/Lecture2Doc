"""Natural-occlusion subset stress test; source-only selection, no official GT.

Clean observations are deliberately withheld, so this is NOT the normal A/B
output. Withheld source images are evaluated only after B has completed.
"""
import argparse,sys,shutil,copy
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write
from lecture2doc.vision import Config,measure_features,small
from lecture2doc.page_restore import restore_page
from lecture2doc.reconstruct import scene_context
from accessmath_run import hardlink_or_copy
from accessmath_evaluate import predicted_ink,score
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-cases',type=int,default=8);a=p.parse_args()
m=read(a.record/'manifest.json');root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
shutil.copytree(a.record/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
folder=root/'results';folder.mkdir(exist_ok=True);config=Config(**m['config']);config.reconstruction=True
frames={f['id']:f for f in m['source_frames']};metrics={v['id']:v for v in m['analysis_metrics']}
contexts=scene_context(root,folder,m['board_states'],frames,config)
states=[s for s in m['board_states'] if len(s['source_frame_ids'])>=12 and 'unconfirmed_change' not in s['quality_flags']
    and max(metrics[f]['occlusion'] for f in s['source_frame_ids'])-min(metrics[f]['occlusion'] for f in s['source_frame_ids'])>.025]
states=sorted(states,key=lambda s:s['end']-s['start'],reverse=True)[:a.max_cases];cases=[]
for state in states:
    ids=state['source_frame_ids'];sample=[ids[j] for j in np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int)]
    data={fid:measure_features(small(cv2.imread(str(root/frames[fid]['image'])),config),config.board) for fid in sample}
    h,w=next(iter(data.values()))['ink'].shape;board=np.zeros((h,w),bool);x,y,x2,y2=m['page_policy']['source_inferred_board_bounds'];board[y:y2,x:x2]=True
    votes=sum(d['ink'].astype('uint16') for d in data.values());vis=sum((~d['foreground']).astype('uint16') for d in data.values())
    support=(votes>=2)&(votes>=vis*.4)&board
    hidden={fid:int((support&d['foreground']).sum()) for fid,d in data.items()}
    eligible=[fid for fid,d in data.items() if (d['foreground']&board).sum()/board.sum()>.025 and hidden[fid]>=20]
    if len(eligible)<4:continue
    anchor=min(eligible,key=lambda fid:(hidden[fid],(data[fid]['foreground']&board).sum()))
    page=dict(id='p'+state['id'][1:],selected_frame_id=anchor,source_frame_ids=eligible,selection_candidates=[],quality_flags=[])
    r=restore_page(root,folder,page,frames,config,m['page_policy'],[state],contexts.get(state['scene']))
    case=dict(state_id=state['id'],interval=[state['start'],state['end']],anchor=anchor,eligible=eligible,result=r)
    # Withheld, low-occlusion observations never participate in restoration.
    clear=[fid for fid in sample if fid not in eligible and abs(frames[fid]['time']-frames[anchor]['time'])<=12]
    ref=min(clear,key=lambda fid:hidden[fid]) if clear else None
    images=[cv2.imread(str(root/frames[anchor]['image'])),cv2.imread(str(root/r['image']))]
    labels=['A: best occluded source','B: '+r['status']]
    if ref:
        reference=cv2.imread(str(root/frames[ref]['image']));images.append(reference);labels.append('Withheld source reference (not official GT)')
        shape=reference.shape[:2];valid=cv2.resize(board.astype('uint8'),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
        truth=predicted_ink(reference,shape,10)&valid
        case['reference_frame_id']=ref;case['reference_occluded_fraction']=float((data[ref]['foreground']&board).sum()/board.sum())
        case['metrics']={label:score(predicted_ink(im,shape,10),truth,valid) for label,im in zip(('A','B'),images[:2])}
    sheet=Image.new('RGB',(480*len(images),300),'white');draw=ImageDraw.Draw(sheet)
    for j,(im,label) in enumerate(zip(images,labels)):
        sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,30));draw.text((j*480+4,6),label,fill='black')
    path=root/(state['id']+'.jpg');sheet.save(path,quality=95);case['comparison']=path.name;cases.append(case)
    write(root/'challenge.json',dict(policy='Real occluded sources only; same microstate; normal time/alignment/consensus gates. Clean references withheld; no official GT.',cases=cases))
    print(state['id'],r['status'],r['reason'],case.get('metrics'),flush=True)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>실제 가림 관측 B 스트레스 테스트</title><h1>실제 가림 원본만 사용한 B 스트레스 테스트</h1><p>정상 A/B와 별도 실험. 깨끗한 관측을 의도적으로 제외했으며 공식 GT가 아닌 보류 원본과 비교합니다.</p>'+''.join(f'<h2>{c["state_id"]} {c["result"]["status"]}: {c["result"]["reason"]}</h2><img style="max-width:100%" src="{c["comparison"]}"><pre>{c.get("metrics",{})}</pre>' for c in cases))
print('cases',len(cases),flush=True)
