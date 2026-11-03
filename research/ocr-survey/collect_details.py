"""Pin selected implementation files and model cards; never download weights."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
index=json.loads((ROOT/'source-index.json').read_text())
EXTRA={'opendatalab/UniMERNet':['unimernet/models/unimernet/__init__.py','unimernet/models/unimernet/encoder_decoder.py','unimernet/models/unimernet/processor.py'], 'BFlameSwift/Uni-MuMER':['scripts/vllm_infer.py','scripts/eval_metrics_calculator.py'], 'datalab-to/surya':['surya/inference/backends/llamacpp.py','surya/inference/parsers.py','surya/inference/prompts.py']}
MODELS=['microsoft/trocr-base-handwritten','PaddlePaddle/PP-OCRv6_small_rec','PaddlePaddle/korean_PP-OCRv5_mobile_rec','PaddlePaddle/PP-FormulaNet_plus-M','wanderkid/unimernet_small','phxember/Uni-MuMER-Qwen3-VL-2B','PaddlePaddle/PaddleOCR-VL-1.6','zai-org/GLM-OCR','lightonai/LightOnOCR-2-1B','lightonai/LightOnOCR-3-4B','Qwen/Qwen3-VL-2B-Instruct']
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Lecture2Doc-OCR-Survey'})
 with urllib.request.urlopen(req,timeout=40) as r:data=r.read(2*1024*1024+1)
 if len(data)>2*1024*1024:raise ValueError('Unexpected size')
 return data
def extra(repo,paths):
 row=next(x for x in index if x['repo']==repo);sha=row['sha'];dest=ROOT/'sources'/repo.replace('/','--')/sha
 for path in paths:
  url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}';data=get(url);p=dest/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  if not any(f['path']==path for f in row['files']):row['files'].append(dict(path=path,url=url,local_path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest()))
 return repo
cards=[]
def card(model):
 meta=json.loads(get('https://huggingface.co/api/models/'+model));sha=meta['sha'];url=f'https://huggingface.co/{model}/raw/{sha}/README.md';data=get(url);dest=ROOT/'model-cards'/model.replace('/','--')/sha;dest.mkdir(parents=True,exist_ok=True);(dest/'README.md').write_bytes(data)
 c=meta.get('cardData') or {};return dict(model=model,revision=sha,checked_date='2026-11-02',license=c.get('license'),languages=c.get('language'),url=url,local_path=str((dest/'README.md').relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 jobs={pool.submit(extra,repo,paths):('code',repo) for repo,paths in EXTRA.items()};jobs.update({pool.submit(card,m):('card',m) for m in MODELS})
 for f in concurrent.futures.as_completed(jobs):
  kind,name=jobs[f]
  try:
   v=f.result()
   if kind=='card':cards.append(v)
   print(kind,name,'OK',flush=True)
  except Exception as e:print(kind,name,type(e).__name__,str(e),flush=True);cards.append(dict(model=name,error=str(e)))
(ROOT/'source-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2));(ROOT/'model-card-index.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2))
