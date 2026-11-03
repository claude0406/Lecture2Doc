"""Actual video diagnostic: rescue the historical occluded 616s source.

The deliberately overridden A anchor makes this distinct from the normal A/B
run, whose A already selects a clean source. GT is read only after restoration.
"""
import sys,copy,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write
from lecture2doc.vision import Config
from lecture2doc.page_restore import restore_page
from lecture2doc.reconstruct import scene_context
from accessmath_run import hardlink_or_copy
from accessmath_evaluate import annotations,predicted_ink,score
src=Path('experiments/board-pages/lecture01');root=Path('experiments/page-restoration/actual-source-rescue');root.mkdir(parents=True,exist_ok=True)
m=read(src/'manifest.json');frames={f['id']:f for f in m['source_frames']}
shutil.copytree(src/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
page=copy.deepcopy(next(p for p in m['board_pages'] if p['start']<616.1155<p['end']))
anchor=min(frames.values(),key=lambda f:abs(f['time']-616.1155));page['selected_frame_id']=anchor['id'];page['selection_candidates']=[]
target=src/m['page_diagnostics']/(page['id']+'.target.png');dest=root/'terminal.png'
if not dest.exists():hardlink_or_copy(target,dest)
page['terminal_ink_mask']=dest.name
folder=root/'b';folder.mkdir(exist_ok=True);config=Config(**m['config']);config.reconstruction=True
contexts=scene_context(root,folder,m['board_states'],frames,config)
r=restore_page(root,folder,page,frames,config,m['page_policy'],m['board_states'],contexts.get(page['scene']))
# Evaluation only: official timestamp chooses the reference, never the donor.
keys,_=annotations(Path('datasets/accessmath/annotations/data/annotations/AccessMath2015_lecture_01'))
k=min(keys,key=lambda k:abs(k['time']-607.674))
images=[k['image'],cv2.imread(str(root/anchor['image'])),cv2.imread(str(root/r['image']))]
labels=['GT (evaluation only)',f"Historical anchor {anchor['time']:.3f}s",'B: '+r['reason']]
metrics={name:score(predicted_ink(im,k['ink'].shape,10),k['ink'],k['valid']) for name,im in zip(('GT','A_anchor','B'),images)}
sheet=Image.new('RGB',(1440,300),'white');draw=ImageDraw.Draw(sheet)
for j,(im,label) in enumerate(zip(images,labels)):
 sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,30));draw.text((j*480+4,5),label,fill='black')
sheet.save(root/'comparison.jpg',quality=95)
write(root/'result.json',dict(normal_A_B_run=False,selection_uses_ground_truth=False,anchor_frame_id=anchor['id'],result=r,metrics=metrics))
print(r,metrics,flush=True)
