"""Reproducible synthetic video: no source frame exposes the whole board."""
import sys
from pathlib import Path
import cv2
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from fixtures import board
from lecture2doc.pipeline import process
from lecture2doc.vision import Config
root=Path('experiments/page-restoration/synthetic-complementary');root.mkdir(parents=True,exist_ok=True)
truth=board(2,white=True)
cv2.putText(truth,'x+y = 3',(850,300),cv2.FONT_HERSHEY_SIMPLEX,1.6,(25,25,25),3)
for x,y in ((25,55),(1040,55),(1040,660)):cv2.putText(truth,'B1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(90,90,90),2)
cv2.imwrite(str(root/'truth.png'),truth)
video=root/'two-occlusions.mp4'
if not video.exists():
 writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),4,(1280,720))
 for i in range(64):
  im=truth.copy();x=180 if int(i/8)%2==0 else 900
  cv2.rectangle(im,(x,150),(x+150,710),(115,101,85),-1);cv2.circle(im,(x+75,130),60,(160,160,190),-1)
  writer.write(im)
 writer.release()
m=process(video,root/'record',Config(),progress=lambda s:print(s,flush=True))
print([(r['status'],r['reason'],r.get('estimated_added_ink_pixels')) for r in m['page_reconstructions']],flush=True)
