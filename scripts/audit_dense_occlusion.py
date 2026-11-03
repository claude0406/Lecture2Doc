"""Audit every native frame of frozen real intervals, after restoration.

Outputs are diagnostic only, never fed back into A/B. No annotation input.
"""
import argparse,sys
from pathlib import Path
import av,cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write
from lecture2doc.vision import Config,small,measure_features
p=argparse.ArgumentParser();p.add_argument('--intervals',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--cases',type=int,nargs='+',required=True);a=p.parse_args()
r=read(a.intervals/'results.json');m=read(a.record/'manifest.json');config=Config(**m['config']);rows=[]
for number in a.cases:
 case=r['results'][number];q=case['proposal']
 target=cv2.imread(str(a.record/q['parent_target']),0)>0
 board=cv2.imread(str(a.record/q['parent_board']),0)>0
 target &=board
 best=None;counts=[]
 with av.open(str(a.record/m['source']['path'])) as c:
  stream=c.streams.video[0];origin=float((stream.start_time or 0)*stream.time_base)
  c.seek(int((q['start']+origin)/stream.time_base),stream=stream)
  for f in c.decode(stream):
   t=float(f.time)-origin
   if t<q['start']:continue
   if t>=q['end']:break
   image=f.to_ndarray(format='bgr24');v=measure_features(small(image,config),config.board)
   hidden=int((target&v['foreground']).sum());missing=int((target&~cv2.dilate(v['ink'].astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)&~v['foreground']).sum())
   counts.append(dict(time=t,hidden_known_ink_pixels=hidden,missing_visible_known_ink_pixels=missing))
   if best is None or (hidden,missing)<best[0]:best=((hidden,missing),t,image)
 row=dict(case=number,start=q['start'],end=q['end'],frames=len(counts),known_ink_pixels=int(target.sum()),best_time=best[1],best_hidden_pixels=best[0][0],best_missing_visible_pixels=best[0][1],observations=counts)
 cv2.imwrite(str(a.intervals/f'dense-case-{number:02d}-best.jpg'),best[2]);rows.append(row)
 write(a.intervals/'dense-audit.json',dict(uses_gt=False,used_as_algorithm_input=False,caveat='Foreground heuristic on every decoded frame; absence of a completely clear frame is not a human segmentation proof.',results=rows))
 print({k:v for k,v in row.items() if k!='observations'},flush=True)
