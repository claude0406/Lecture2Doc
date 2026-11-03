"""Paired re-run of frozen natural-occlusion inputs; references are evaluation only."""
import argparse,sys,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write,digest
from lecture2doc.vision import Config
from lecture2doc.page_restore import restore_page
from lecture2doc.reconstruct import scene_context
from accessmath_run import hardlink_or_copy
from accessmath_evaluate import predicted_ink,score
p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
m=read(a.record/'manifest.json');prior=read(a.frozen/'challenge.json');root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
shutil.copytree(a.record/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
config=Config(**m['config']);config.reconstruction=True;frames={f['id']:f for f in m['source_frames']}
folder=root/'results';folder.mkdir(exist_ok=True);contexts=scene_context(root,folder,m['board_states'],frames,config);cases=[]
for old in prior['cases']:
 state=next(s for s in m['board_states'] if s['id']==old['state_id'])
 page=dict(id='p'+state['id'][1:],selected_frame_id=old['anchor'],source_frame_ids=old['eligible'],selection_candidates=[],quality_flags=[])
 r=restore_page(root,folder,page,frames,config,m['page_policy'],[state],contexts.get(state['scene']))
 row=dict(state_id=state['id'],anchor=old['anchor'],eligible=old['eligible'],result=r,previous_result=old['result'])
 images=[cv2.imread(str(root/frames[old['anchor']]['image'])),cv2.imread(str(root/r['image']))];labels=['Frozen actual A','New B: '+r['status']]
 ref=old.get('reference_frame_id')
 if ref:
  reference=cv2.imread(str(root/frames[ref]['image']));images.append(reference);labels.append('Frozen reference (not official GT)')
  H,W=reference.shape[:2];valid=np.zeros((H,W),bool);x,y,x2,y2=m['page_policy']['source_inferred_board_bounds'];valid[round(y*H/360):round(y2*H/360),round(x*W/640):round(x2*W/640)]=True
  truth=predicted_ink(reference,(H,W),10)&valid
  row['reference_frame_id']=ref;row['metrics']={name:score(predicted_ink(im,(H,W),10),truth,valid) for name,im in zip(('A','B'),images[:2])}
 sheet=Image.new('RGB',(480*len(images),300),'white');draw=ImageDraw.Draw(sheet)
 for j,(im,label) in enumerate(zip(images,labels)):
  sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(480*j,28));draw.text((480*j+4,5),label,fill='black')
 sheet.save(root/(state['id']+'.jpg'),quality=95);row['comparison']=state['id']+'.jpg';cases.append(row)
 write(root/'challenge.json',dict(frozen_input=str(a.frozen),uses_ground_truth_for_selection=False,source_hashes={str(f):digest(f) for f in (Path('lecture2doc/vision.py'),Path('lecture2doc/page_restore.py'))},cases=cases))
 print(state['id'],r['status'],r['reason'],row.get('metrics'),flush=True)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>固定実映像入力 B比較</title><h1>고정한 실제 가림 입력의 B 전후 비교</h1><p>원본·기준 시각·후보·참고 이미지는 이전 실험과 동일합니다. 공식 GT 점수가 아닙니다.</p>'+''.join(f'<h2>{c["state_id"]}: {c["result"]["status"]} — {c["result"]["reason"]}</h2><img style="max-width:100%" src="{c["comparison"]}"><pre>{c.get("metrics",{})}</pre>' for c in cases))
