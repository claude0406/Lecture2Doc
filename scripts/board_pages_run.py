"""Re-evaluate A from immutable decoded observations. Never reads annotations."""
import argparse,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,stage
from lecture2doc.pages import build_pages
from lecture2doc.vision import Config
from accessmath_run import hardlink_or_copy
p=argparse.ArgumentParser();p.add_argument('--from-record',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();src=a.from_record.resolve();out=a.out.resolve()
if src==out: raise ValueError('Use a new experiment directory')
out.mkdir(parents=True,exist_ok=True)
m=read(src/'manifest.json')
if (src/'source.json').exists() and not (out/'source.json').exists():
    shutil.copy2(src/'source.json',out/'source.json')
for name in (m['source']['path'],m['playback']):
    if (src/name).exists():
        (out/name).parent.mkdir(parents=True,exist_ok=True)
        if not (out/name).exists():hardlink_or_copy(src/name,out/name)
shutil.copytree(src/'cache/decode',out/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
c=Config(**m['config']);conf=c.json();conf.pop('reconstruction');conf.pop('tile_size')
result=stage(out,'pages',[m['source']['sha256'],conf],lambda folder:build_pages(out,folder,m['source_frames'],c,m['source']['duration'],m['analysis_metrics'],lambda s:print(s,flush=True)))
m['board_pages']=result['pages'];m['page_events']=result['events'];m['page_policy']=result['policy']
for page in m['board_pages']:
    page['transcript_ids']=[t['id'] for t in m['transcript']['segments'] if t['end']>page['start'] and t['start']<page['end']]
# Historical microstate B images refer to the old immutable record; page mode uses only originals.
m['reconstructed_boards']=[]
m['parent_record']=str(src)
m['primary_output']='board_pages'
m['page_reconstruction_status']='not_implemented'
if 'persistence_can_miss_short_lived_writing' not in m['quality_flags']:
    m['quality_flags'].append('persistence_can_miss_short_lived_writing')
write(out/'manifest.json',m)
print('PAGES',len(m['board_pages']),flush=True)
for page in m['board_pages']:
    print(page['id'],round(page['start'],2),round(page['end'],2),page['selection_candidates'][0],page['quality_flags'],flush=True)
