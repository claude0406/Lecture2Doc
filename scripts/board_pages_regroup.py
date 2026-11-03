"""Reuse measured source features for reviewable page-policy iteration."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,digest
from lecture2doc.pages import observations,group_pages
from lecture2doc.vision import Config
p=argparse.ArgumentParser();p.add_argument('record',type=Path);a=p.parse_args();r=a.record
m=read(r/'manifest.json');folder=next((r/'cache/pages').iterdir());c=Config(**m['config'])
data=observations(r,folder,m['source_frames'],c)
rev=r/'page-decisions'/digest(Path('lecture2doc/pages.py'))[:12];rev.mkdir(parents=True,exist_ok=True)
result=group_pages(m['source_frames'],data,c,m['source']['duration'],m['analysis_metrics'],rev)
write(rev/'result.json',result)
m.update(primary_output='board_pages',page_reconstruction_status='not_run',board_pages=result['pages'],page_events=result['events'],page_policy=result['policy'],page_diagnostics=str(rev.relative_to(r)))
for page in m['board_pages']:
 for field,suffix in [('terminal_ink_mask','.target.png'),('analysis_board_mask','.board.png'),('terminal_absence_evidence','.evidence.npz')]:
  page[field]=(rev/(page['id']+suffix)).relative_to(r).as_posix()
 page['transcript_ids']=[t['id'] for t in m['transcript']['segments'] if t['end']>page['start'] and t['start']<page['end']]
if 'persistence_can_miss_short_lived_writing' not in m['quality_flags']:
 m['quality_flags'].append('persistence_can_miss_short_lived_writing')
write(r/'manifest.json',m)
print('PAGES',len(m['board_pages']))
for page in m['board_pages']:
 for field,suffix in [('terminal_ink_mask','.target.png'),('analysis_board_mask','.board.png'),('terminal_absence_evidence','.evidence.npz')]:
  page[field]=(rev/(page['id']+suffix)).relative_to(r).as_posix()
 print(page['id'],round(page['start'],2),round(page['end'],2),page['selection_candidates'][0],page['quality_flags'])
