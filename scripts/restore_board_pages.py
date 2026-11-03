"""Run B on a preserved A record into a new directory; no annotation input."""
import argparse,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,stage,digest
from lecture2doc.vision import Config
from lecture2doc.page_restore import restore_pages
from accessmath_run import hardlink_or_copy
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
src=a.record.resolve();root=a.out.resolve()
if root==src:raise ValueError('Preserve A: use a new output directory')
root.mkdir(parents=True,exist_ok=True)
m=read(src/'manifest.json')
for existing in (root/'manifest.json',root/'source.json'):
 if existing.exists():
  previous=read(existing);previous_hash=previous.get('sha256') or previous.get('source',{}).get('sha256')
  if previous_hash!=m['source']['sha256']:raise ValueError('Output directory contains a different video')
for path in {m['source']['path'],m['playback']}:
 (root/path).parent.mkdir(parents=True,exist_ok=True)
 if not (root/path).exists():hardlink_or_copy(src/path,root/path)
shutil.copytree(src/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
if m.get('page_diagnostics') and (src/m['page_diagnostics']).exists():
 shutil.copytree(src/m['page_diagnostics'],root/m['page_diagnostics'],dirs_exist_ok=True,copy_function=hardlink_or_copy)
if (src/'source.json').exists():shutil.copy2(src/'source.json',root/'source.json')
# Preserve A's observed terminal target as a content constraint for donors.
for page in m['board_pages']:
 for field,suffix in [('terminal_ink_mask','.target.png'),('analysis_board_mask','.board.png'),('terminal_absence_evidence','.evidence.npz')]:
    target=None
    if page.get(field):target=src/page[field]
    elif m.get('page_diagnostics'):target=src/m['page_diagnostics']/(page['id']+suffix)
    else:
        for cached in (src/'cache/pages').glob('*/stage.json'):
            old=read(cached)
            if old.get('policy',{}).get('version')!=m.get('page_policy',{}).get('version'):continue
            if any(p['id']==page['id'] and p['selected_frame_id']==page['selected_frame_id'] and p['start']==page['start'] and p['end']==page['end'] for p in old.get('pages',[])):
                target=cached.parent/(page['id']+suffix);break
    if target is not None and target.exists():
        dest=root/'page-targets'/(page['id']+'-'+field+'-'+digest(target)[:12]+suffix);dest.parent.mkdir(exist_ok=True)
        if not dest.exists():hardlink_or_copy(target,dest)
        page[field]=dest.relative_to(root).as_posix()
config=Config(**m['config']);config.reconstruction=True
r=stage(root,'page-restoration',[m['source']['sha256'],config.json(),m['board_pages'],m['page_policy'],m['board_states']],
 lambda folder:restore_pages(root,folder,m['board_pages'],m['source_frames'],config,m['page_policy'],m['board_states'],lambda s:print(s,flush=True)))
m.update(page_reconstructions=r,page_reconstruction_status='complete',config=config.json(),parent_record=str(src))
# Legacy microstate B artifacts are not exposed as page B results.
m['reconstructed_boards']=[]
write(root/'manifest.json',m)
write(root/'algorithm.json',dict(uses_ground_truth=False,sources={str(f):digest(f) for f in [Path('lecture2doc/page_restore.py'),Path('lecture2doc/reconstruct.py'),Path('lecture2doc/vision.py'),Path('lecture2doc/pages.py')]}))
from collections import Counter
print(dict(Counter(x['status'] for x in r)),flush=True)
