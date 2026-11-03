"""Re-evaluate B on frozen real intervals, including the exact same A evidence."""
import argparse,sys,shutil,copy
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write,digest
from lecture2doc.page_restore import restore_page
from lecture2doc.vision import Config
from accessmath_run import hardlink_or_copy
from accessmath_evaluate import predicted_ink,score
p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
prior=read(a.frozen/'results.json');m=read(a.record/'manifest.json');root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
shutil.copytree(a.record/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
frames={f['id']:f for f in m['source_frames']};config=Config(**m['config']);config.reconstruction=True;rows=[]
for number,old in enumerate(prior['results']):
 if 'page' not in old:continue
 page=copy.deepcopy(old['page']);proposal=old['proposal']
 for field in ['terminal_ink_mask','analysis_board_mask','terminal_absence_evidence']:
  if page.get(field):
   target=root/page[field];target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():hardlink_or_copy(a.frozen/page[field],target)
 folder=root/f'case-{number:02d}';folder.mkdir(exist_ok=True)
 state=dict(id='interval',scene=0,start=proposal['start'],end=proposal['end'],source_frame_ids=proposal['frame_ids'],quality_flags=[],boundary_reason='end_of_interval')
 r=restore_page(root,folder,page,frames,config,m['page_policy'],[state])
 reference=cv2.imread(str(a.record/frames[proposal['reference_frame_id']]['image']))
 A=cv2.imread(str(root/frames[page['selected_frame_id']]['image']));B=cv2.imread(str(root/r['image']))
 board=cv2.imread(str(a.record/proposal['parent_board']),0)>0
 H,W=reference.shape[:2];valid=cv2.resize(board.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 truth=predicted_ink(reference,(H,W),10)&valid
 metrics={label:score(predicted_ink(im,(H,W),10),truth,valid) for label,im in [('A',A),('B',B)]}
 row=dict(proposal=proposal,page=page,result=r,previous_result=old['result'],metrics=metrics)
 images=[A,B,reference];labels=[f"Frozen actual A {proposal['start']:.1f}-{proposal['end']:.1f}s",'New B: '+r['status'],'Outside reference (not official GT)']
 sheet=Image.new('RGB',(1440,300),'white');draw=ImageDraw.Draw(sheet)
 for j,(im,label) in enumerate(zip(images,labels)):
  sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,28));draw.text((j*480+4,5),label,fill='black')
 sheet.save(folder/'comparison.jpg',quality=95);row['comparison']=(folder/'comparison.jpg').relative_to(root).as_posix();rows.append(row)
 write(root/'results.json',dict(frozen_input=str(a.frozen),record=str(a.record),uses_gt=False,clear_reference_withheld=True,source_hashes={str(p):digest(p) for p in [Path('lecture2doc/page_restore.py'),Path('lecture2doc/vision.py')]},results=rows))
 print(number,proposal['start'],r['status'],r['reason'],metrics,flush=True)
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>고정 실제 관측·A 근거에서 B 재평가</h1><p>공식 GT가 아닌 보류 원본과 비교합니다.</p>'+''.join(f'<h2>{r["result"]["status"]}: {r["result"]["reason"]}</h2><img style="max-width:100%" src="{r["comparison"]}"><pre>{r["metrics"]}</pre>' for r in rows))
