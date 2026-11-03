"""Manual transcription audit on fixed real-board regions; no GT enters OCR."""
import html
import json
from pathlib import Path
import re
import unicodedata
import cv2

ROOT = Path(__file__).resolve().parent
SAMPLES = [
 ('lecture01',0,3,'① Geometrical Interpretation'),
 ('lecture01',0,4,'② Algebraic Interpretation.'),
 ('lecture01',0,5,'System of Linear equations.'),
 ('lecture01',0,8,'What is represented by this equation ?'),
 ('lecture01',0,10,'1-dim'),('lecture01',0,11,'3rd-dim'),
 ('lecture01',0,12,'plane'),('lecture01',0,15,'2-dim'),
 ('lecture01',3,1,'UNIQUE'),('lecture01',3,2,'PARALLEL lines'),
 ('lecture01',3,3,'solution'),('lecture01',3,4,'no solution.'),
 ('lecture02-testing',0,1,'Lecture 2 (mimio)'),('lecture02-testing',0,2,'Recall:'),
 ('lecture02-testing',0,3,'1) System of linear Equations'),
 ('lecture02-testing',0,4,'2) Number of solutions'),
 ('lecture02-testing',0,5,'3) Row picture &'),('lecture02-testing',0,6,'Column picture'),
 ('lecture02-testing',5,1,'at least one solution'),('lecture02-testing',5,2,'Consistent system'),
 ('lecture02-testing',5,3,'→ no solution'),('lecture02-testing',5,4,'Inconsistent system'),
 ('lecture02-testing',5,9,'Multiplication'),('lecture02-testing',5,10,'by a non-zero'),
 ('lecture02-testing',5,11,'scalar'),
]

def normalize(s):
 return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip()

def edits(a,b):
 previous=list(range(len(b)+1))
 for i, x in enumerate(a,1):
  current=[i]
  for j,y in enumerate(b,1):
   current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
  previous=current
 return previous[-1]

rows=[]
for name, page_index, order, gt in SAMPLES:
 m=json.loads((ROOT/name/'manifest.json').read_text())
 p=m['ocr']['pages'][page_index]
 r=p['regions'][order-1]
 rows.append(dict(record=name,page_id=p['board_page_id'],input=p['input'],bbox=r['bbox'],
                  polygon=r['polygon'],image=f'../{name}/{r["image"]}',gt=gt,prediction=r['text'],score=r['score']))
# Explicit missed sentence: not represented by any RGB OCR region on this page.
m=json.loads((ROOT/'lecture01/manifest.json').read_text())
p=m['ocr']['pages'][3]
im=cv2.imread(str(ROOT/'lecture01'/p['input']['image']))
box=[235,300,600,100]
cv2.imwrite(str(ROOT/'review/missed-line.jpg'),im[300:400,235:835])
rows.append(dict(record='lecture01',page_id=p['board_page_id'],input=p['input'],bbox=box,
                 image='missed-line.jpg',gt='two lines ⇒ two linear equations',prediction='',score=None))
summary={}
for name in ['lecture01','lecture02-testing','all']:
 subset=[r for r in rows if name=='all' or r['record']==name]
 ce=sum(edits(normalize(r['gt']),normalize(r['prediction'])) for r in subset)
 chars=sum(len(normalize(r['gt'])) for r in subset)
 we=sum(edits(normalize(r['gt']).split(),normalize(r['prediction']).split()) for r in subset)
 words=sum(len(normalize(r['gt']).split()) for r in subset)
 summary[name]=dict(regions=len(subset),character_edits=ce,characters=chars,cer=ce/chars,
                    word_edits=we,words=words,wer=we/words,
                    exact=sum(normalize(r['gt'])==normalize(r['prediction']) for r in subset))
result=dict(scope='Manual audit of 26 selected text regions on four real pages; includes one missed line.',
            normalization='Unicode NFC, whitespace collapsed/trimmed; case and punctuation preserved.',
            selection='Post-run diagnostic sample; 25 detected regions plus one manually located missed sentence.',
            summary=summary,regions=rows)
(ROOT/'ocr-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
out=['<!doctype html><meta charset="utf-8"><title>V1 OCR 실제 판서 비교</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;color:#19312e;background:#f5f7f2}table{width:100%;border-collapse:collapse}td,th{padding:14px;border:1px solid #d8e0da;text-align:left;background:white}img{max-width:380px;max-height:110px}small{color:#60716c}.bad{background:#fff2df}</style><h1>V1 · 실제 판서 OCR</h1><p>Lecture 01·02의 네 판서에서 고른 문장·라벨 26곳. 한 줄 미검출을 포함합니다. 대소문자·구두점을 유지하고 공백만 정리해 비교했습니다.</p>']
s=summary['all']
out.append(f'<p>CER {s["cer"]:.2%} · WER {s["wer"]:.2%} · 전체 일치 {s["exact"]}/{s["regions"]}</p><table><tr><th>원본 영역</th><th>사람 전사</th><th>OCR</th></tr>')
for r in rows:
 cls='' if normalize(r['gt'])==normalize(r['prediction']) else ' class="bad"'
 out.append(f'<tr><td><small>{r["record"]} · {r["page_id"]}</small><br><img src="{html.escape(r["image"])}"></td><td>{html.escape(r["gt"])}</td><td{cls}>{html.escape(r["prediction"]) or "(미검출)"}</td></tr>')
out.append('</table><h2>같은 판서의 전처리 비교</h2><p>왼쪽 RGB / 오른쪽 적응 이진화. 이진화에서 추가되는 오류와 누락을 함께 확인할 수 있습니다.</p>')
for name in ['lecture01','lecture01-adaptive']:
 m=json.loads((ROOT/name/'manifest.json').read_text());p=m['ocr']['pages'][0]
 out.append(f'<a href="../{name}/{p["overlay"]}"><img style="max-width:48%;max-height:none" src="../{name}/{p["overlay"]}"></a>')
(ROOT/'review/index.html').write_text(''.join(out))
print(json.dumps(summary,ensure_ascii=False,indent=2))
