"""Fetch small, pinned official source files for a read-only C-stage survey.

No model weights, packages, inference, or application changes.
"""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPOS={
 'PaddlePaddle/PaddleOCR':['README_en.md','LICENSE','paddleocr/_pipelines/ocr.py','paddleocr/_pipelines/formula_recognition.py','paddleocr/_pipelines/paddleocr_vl.py','docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md'],
 'opendatalab/UniMERNet':['README.md','LICENSE','demo.py','pyproject.toml','unimernet/models/unimernet.py'],
 'BFlameSwift/Uni-MuMER':['README.md','LICENSE','requirements.txt'],
 'lukas-blecher/LaTeX-OCR':['README.md','LICENSE','pix2tex/cli.py','setup.cfg'],
 'datalab-to/surya':['README.md','LICENSE','MODEL_LICENSE','pyproject.toml'],
 'zai-org/GLM-OCR':['README.md','LICENSE','pyproject.toml'],
}
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Lecture2Doc-OCR-Survey','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
 with urllib.request.urlopen(req,timeout=40) as r:
  data=r.read(8*1024*1024+1)
 if len(data)>8*1024*1024:raise ValueError('Unexpected large survey source')
 return data
def collect(repo,paths):
 commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/HEAD'));sha=commit['sha']
 tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'));files={x['path'] for x in tree['tree'] if x['type']=='blob'}
 dest=ROOT/'sources'/repo.replace('/','--')/sha;dest.mkdir(parents=True,exist_ok=True)
 candidates=sorted(p for p in files if p.endswith('.py') and any(k in p.lower() for k in ['infer','predict','model','eval']))
 result=dict(repo=repo,sha=sha,commit_date=commit['commit']['committer']['date'],checked_date='2026-11-02',code_candidates=candidates[:60],files=[])
 for p in paths:
  if p not in files:continue
  url=f'https://raw.githubusercontent.com/{repo}/{sha}/{p}';data=get(url);out=dest/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
  result['files'].append(dict(path=p,url=url,local_path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest()))
 print(repo,sha,len(result['files']),flush=True)
 return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 future={pool.submit(collect,repo,paths):repo for repo,paths in REPOS.items()}
 for f in concurrent.futures.as_completed(future):
  try:results.append(f.result())
  except Exception as e:results.append(dict(repo=future[f],error=f'{type(e).__name__}: {e}'));print(future[f],str(e),flush=True)
(ROOT/'source-index.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
