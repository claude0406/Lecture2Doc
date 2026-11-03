"""Native source crops and preprocessing previews."""
import json,sys,html,os
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lecture2doc.storage import read,write,digest
root=Path(__file__).resolve().parent;out=root/'examples';out.mkdir(exist_ok=True)
examples=[('01-formula','lecture01',1,[.06,.25,.50,.66],'Equation and nearby prose'),('01-prose','lecture01',2,[.40,.47,.94,.83],'Handwritten explanation / inline symbols'),('02-matrix','lecture02-testing',2,[.38,.33,.96,.83],'Matrix / reduction arrows / labels'),('02-multiblock','lecture02-testing',15,[.06,.20,.55,.85],'Multiple matrices / bracket grouping'),('06-symbols','lecture06-additional',5,[.35,.20,.68,.85],'Faint writing / coloured vectors'),('01-diagram','lecture01',1,[.53,.14,.93,.58],'Graph with mathematical labels')]
rows=[];panels=[]
for name,lecture,pageno,box,title in examples:
 record=Path('experiments/ab-refinement/accepted')/lecture;m=read(record/'manifest.json');page=m['board_pages'][pageno];f=next(f for f in m['source_frames'] if f['id']==page['selected_frame_id']);source=record/f['image'];im=cv2.imread(str(source));H,W=im.shape[:2];x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[W,H,W,H])];crop=im[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,15)
 paths=[]
 for tag,a in [('rgb',crop),('gray',gray),('binary',binary)]:
  p=out/f'{name}-{tag}.png';cv2.imwrite(str(p),a);paths.append(p)
 panel=Image.new('RGB',(1260,300),'white');d=ImageDraw.Draw(panel)
 for j,p in enumerate(paths):
  a=Image.open(p).convert('RGB');a.thumbnail((408,250));panel.paste(a,(j*420+(420-a.width)//2,42+(250-a.height)//2));d.text((j*420+5,5),title+' / '+['RGB','Gray','Adaptive binary'][j],fill='black')
 panel.save(out/(name+'-comparison.jpg'),quality=95)
 row=dict(id=name,title=title,record=str(record),board_page_id=page['id'],source_frame_id=f['id'],source_time=f['time'],source_image=str(source),source_image_sha256=digest(source),source_box_pixels=[x0,y0,x1,y1],recognition_run=False,ground_truth_annotation=None,variants=[str(p.relative_to(root)) for p in paths]);rows.append(row);panels.append(panel)
write(root/'example-index.json',dict(purpose='Survey examples; no OCR inference, text labels, or accuracy measurement',examples=rows))
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>판서 인식 입력 예시</title><style>body{max-width:1300px;margin:30px auto;font:16px sans-serif}img{max-width:100%}</style><h1>실제 판서 입력 · 전처리 비교</h1><p>왼쪽부터 RGB 원본, 회색조, 적응 이진화입니다.</p><p><a href="../accuracy.html">수식 인식 정확도와 도입 기준</a></p>'+''.join(f'<h2>{html.escape(r["id"])} · {r["source_time"]:.3f}s</h2><img src="{r["id"]}-comparison.jpg">' for r in rows))
contact=Image.new('RGB',(1260,300*len(panels)),'white')
for i,p in enumerate(panels):contact.paste(p,(0,i*300))
contact.save(out/'contact.jpg',quality=94);print(len(rows),'source crops, 3 variants each; no inference')
