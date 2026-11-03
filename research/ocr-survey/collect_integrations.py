"""Pin extra integration candidates without installing or running them."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPOS={'breezedeus/Pix2Text':['README.md','LICENSE','pix2text/text_formula_ocr.py','pix2text/pix_to_text.py','pix2text/latex_ocr.py','setup.py'], 'RapidAI/RapidOCR':['README.md','LICENSE','python/rapidocr/default_models.yaml'], 'OleehyO/TexTeller':['README.md','LICENSE','src/inference.py']}
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Lecture2Doc-OCR-Survey'}),timeout=30) as r:return r.read(2*1024*1024)
def run(repo,paths):
 c=json.loads(get(f'https://api.github.com/repos/{repo}/commits/HEAD'));sha=c['sha'];row=dict(repo=repo,sha=sha,commit_date=c['commit']['committer']['date'],checked_date='2026-11-02',files=[])
 for p in paths:
  url=f'https://raw.githubusercontent.com/{repo}/{sha}/{p}'
  try:data=get(url)
  except Exception:continue
  out=ROOT/'sources'/repo.replace('/','--')/sha/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);row['files'].append(dict(path=p,url=url,local_path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest()))
 print(repo,sha,len(row['files']),flush=True);return row
rows=json.loads((ROOT/'source-index.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 jobs={pool.submit(run,r,p):r for r,p in REPOS.items()}
 for f in concurrent.futures.as_completed(jobs):
  try:
   row=f.result();rows=[r for r in rows if r['repo']!=row['repo']]+[row]
  except Exception as e:print(jobs[f],str(e),flush=True)
(ROOT/'source-index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
