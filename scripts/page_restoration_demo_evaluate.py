"""Evaluate synthetic restoration independently of AccessMath."""
import sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write
from lecture2doc.page_restore import corrected_patch
from accessmath_evaluate import predicted_ink,score
root=Path('experiments/page-restoration/synthetic-complementary');record=root/'record';m=read(record/'manifest.json')
p=m['board_pages'][0];r=m['page_reconstructions'][0];frames={f['id']:f for f in m['source_frames']}
truth=cv2.imread(str(root/'truth.png'));a=cv2.imread(str(record/frames[p['selected_frame_id']]['image']));b=cv2.imread(str(record/r['image']))
shape=truth.shape[:2];valid=np.ones(shape,bool)
metrics={name:{str(t):score(predicted_ink(im,shape,t),predicted_ink(truth,shape,t),valid) for t in (10,20,30)} for name,im in [('A',a),('B',b)]}
prov=read(record/r['provenance']);replay=a.copy()
for patch in prov['patches']:
 x,y,w,h=(patch[k] for k in ('x','y','width','height'))
 im=cv2.imread(str(record/frames[patch['source_frame_id']]['image']))[y:y+h,x:x+w]
 replay[y:y+h,x:x+w]=corrected_patch(im,x,y,w,h,prov['roi_pixel_box'],patch.get('photometric_correction'))
assert np.array_equal(replay,b)
write(root/'evaluation.json',dict(synthetic=True,actual_video_claim=False,page_count=len(m['board_pages']),
 result=r,metrics=metrics,source_provenance_pixel_replay_exact=True,provenance_rectangles=len(prov['patches'])))
sheet=Image.new('RGB',(1440,600),'white');draw=ImageDraw.Draw(sheet)
images=[a,b,truth,cv2.imread(str(record/r['diagnostics'][0]))]
labels=['SYNTHETIC A: every source has occlusion','SYNTHETIC B: composite, no inpainting','Synthetic reference','Green: replaced from source, red: unresolved']
for j,(im,label) in enumerate(zip(images,labels)):
 x=(j%3)*480;y=(j//3)*300
 sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(x,y+28));draw.text((x+4,y+6),label,fill='black')
sheet.save(root/'comparison.jpg',quality=95)
print(metrics,flush=True)
