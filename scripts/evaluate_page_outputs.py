"""Frozen accumulated-page outputs vs official GT; unchanged scoring contract.

GT is read only here, after A and B have produced their manifests.
"""
import argparse,html,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,digest
from accessmath_evaluate import annotations,predicted_ink,score
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--gt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['training','testing'],required=True);p.add_argument('--previous',type=Path);a=p.parse_args()
m=read(a.record/'manifest.json');keys,_=annotations(a.gt);frames={f['id']:f for f in m['source_frames']};out=a.out;out.mkdir(parents=True,exist_ok=True)
old=read(a.previous/'manifest.json') if a.previous else None
rows=[]
for k in keys:
 if k['time']>=m['source']['duration']:continue
 page=next(p for p in m['board_pages'] if p['start']<=k['time']<p['end'])
 selected=frames[page['selected_frame_id']];nearest=min(frames.values(),key=lambda f:abs(f['time']-k['time']))
 r=next(r for r in m['page_reconstructions'] if r['board_page_id']==page['id'])
 images=dict(GT=k['image'],nearest=cv2.imread(str(a.record/nearest['image'])),A=cv2.imread(str(a.record/selected['image'])),B=cv2.imread(str(a.record/r['image'])))
 if old:
  op=next(p for p in old['board_pages'] if p['start']<=k['time']<p['end']);of=next(f for f in old['source_frames'] if f['id']==op['selected_frame_id']);images['previous_A']=cv2.imread(str(a.previous/of['image']))
 metrics={label:{str(t):score(predicted_ink(im,k['ink'].shape,t),k['ink'],k['valid']) for t in (10,20,30)} for label,im in images.items()}
 gray=lambda im:cv2.cvtColor(cv2.resize(im,(960,540)),cv2.COLOR_BGR2GRAY).astype('float32')
 phase,response=cv2.phaseCorrelate(gray(k['image']),gray(images['nearest']))
 labels=list(images)+['GT_mask'];images['GT_mask']=cv2.cvtColor((~k['ink']).astype('uint8')*255,cv2.COLOR_GRAY2BGR)
 sheet=Image.new('RGB',(1440,600),'white');draw=ImageDraw.Draw(sheet)
 for j,label in enumerate(labels):
  x,y=j%3*480,j//3*300;im=images[label]
  sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(x,y+25))
  draw.text((x+5,y+5),label+(f" F1 {metrics[label]['10']['f1']*100:.2f}%" if label in metrics else ''),fill='black')
 name=f"gt-{k['index']}.jpg";sheet.save(out/name,quality=95)
 for label in ['GT','nearest','A','B']:cv2.imwrite(str(out/f"gt-{k['index']}-{label}.png"),images[label])
 rows.append(dict(gt_index=k['index'],gt_time=k['time'],page_id=page['id'],selected_time=selected['time'],nearest_time=nearest['time'],B_status=r['status'],B_reason=r['reason'],metrics=metrics,comparison=name,alignment_half_resolution=dict(dx=phase[0],dy=phase[1],response=response),native_shape=list(images['A'].shape),GT_shape=list(k['image'].shape)))
aggregate={str(t):{label:{metric:float(np.mean([r['metrics'][label][str(t)][metric] for r in rows])) for metric in ['recall','precision','f1']} for label in rows[0]['metrics']} for t in (10,20,30)}
write(out/'evaluation.json',dict(record=str(a.record),split=a.split,pages=len(m['board_pages']),sampled_frames=len(frames),uses_gt_for_selection=False,scoring_unchanged=True,scoring_source_sha256=digest(Path('scripts/accessmath_evaluate.py')),caveat='Existing snapshot pixel score, not official unique-CC metric. Completed pages can contain later additions. Native source is resized to GT dimensions exactly as in previous scoring.',aggregate=aggregate,rows=rows))
table='<table><tr><th>Output</th><th>Recall</th><th>Precision</th><th>F1</th></tr>'+''.join('<tr><td>'+label+'</td>'+''.join(f'<td>{v[k]*100:.2f}%</td>' for k in ['recall','precision','f1'])+'</tr>' for label,v in aggregate['10'].items())+'</table>'
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>고정 A/B 출력 — 공식 GT 스냅샷 진단</h1><p>GT는 평가에만 사용. 기존 점수 계산 함수를 유지했습니다. 강의 전체 판서 요약의 공식 unique-CC 지표는 아닙니다.</p>'+table+''.join(f'<h2>{r["gt_time"]:.3f}s · {r["page_id"]} · 선택 {r["selected_time"]:.3f}s · B {html.escape(r["B_status"])}</h2><img src="{r["comparison"]}">' for r in rows))
print(a.record,len(rows),aggregate['10'],flush=True)
