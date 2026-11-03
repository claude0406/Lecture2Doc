"""Cold per-record cache benchmark, plus a cached repeat. No ASR."""
from pathlib import Path
import json
import platform
import time
import resource
from lecture2doc.pipeline import process
from lecture2doc.vision import Config
from lecture2doc.storage import write

base = Path(__file__).resolve().parent
out = base/'lecture01-cold'
if out.exists():
    raise SystemExit('Cold output already exists; preserve it and choose a new benchmark folder.')
config=Config()
source=Path('experiments/ab-refinement/accepted/lecture01/source.mp4')
started=time.monotonic()

def progress(message):
    print(f'{time.monotonic()-started:8.2f}s {message}',flush=True)

for mode in ['cold','warm']:
    start=time.monotonic()
    result=process(source,out,config,ocr_language='en',progress=progress)
    wall=time.monotonic()-start
    if result.get('visual_error') or result['ocr']['status']!='complete':
        raise RuntimeError('Incomplete benchmark: '+str(result.get('visual_error') or result['ocr']))
    evidence=dict(mode=mode,wall_seconds=wall,pipeline_seconds=result['elapsed_seconds'],
                  duration_seconds=result['source']['duration'],frames=len(result['source_frames']),
                  pages=len(result['board_pages']),ocr_regions=sum(len(p['regions']) for p in result['ocr']['pages']),
                  timings=result['timings'],ocr_cache_hits=result['ocr']['cache_hits'],
                  platform=platform.platform(),cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  model_download_included=False,asr=False,config=config.json())
    write(base/f'{mode}.json',evidence)
    write(base/f'{mode}-manifest.json',result)
    print(json.dumps(evidence,ensure_ascii=False),flush=True)
