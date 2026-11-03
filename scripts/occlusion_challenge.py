"""Source-only natural-occlusion subset challenge; GT is not read here.

All qualifying application states are considered. Clean observations are excluded
to ask whether real occluded observations can recover one another. This is a
separate stress test, never the normal full-video score.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write
from lecture2doc.vision import Config,small,measure_features,rank_candidates
from lecture2doc.reconstruct import reconstruct,scene_context
from accessmath_run import hardlink_or_copy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--anchor',choices=['ranked','max_occlusion'],default='ranked')
    p.add_argument('--max-cases',type=int,default=0,help='Source-only longest-state selection, before scoring')
    args=p.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    original=read(args.record/'manifest.json');m=copy.deepcopy(original)
    hardlink_or_copy(args.record/original['source']['path'],root/original['source']['path'])
    shutil.copytree(args.record/'cache/decode',root/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
    folder=root/'results';folder.mkdir(exist_ok=True)
    config=Config(**m['config']);frames={f['id']:f for f in m['source_frames']}
    metrics={x['id']:x for x in m['analysis_metrics']}
    contexts=scene_context(root,folder,m['board_states'],frames,config)
    allowed=None
    if args.max_cases:
        possible=[s for s in m['board_states'] if len(s['source_frame_ids'])>=12
                  and max(metrics[f]['occlusion'] for f in s['source_frame_ids'])-min(metrics[f]['occlusion'] for f in s['source_frame_ids'])>=.025]
        allowed={s['id'] for s in sorted(possible,key=lambda s:s['end']-s['start'],reverse=True)[:args.max_cases]}
    results=[];cases=[]
    for state in m['board_states']:
        ids=state['source_frame_ids'];ctx=contexts.get(state['scene'])
        fallback=dict(id='r'+state['id'][1:],board_state_id=state['id'],status='fallback',
                      image=frames[state['selected_frame_id']]['image'],source_frame_ids=[state['selected_frame_id']],
                      provenance=None,coverage=None,quality_flags=['not_an_occlusion_challenge'])
        scores=[metrics[f]['occlusion'] for f in ids]
        if (allowed is not None and state['id'] not in allowed) or len(ids)<12 or max(scores)-min(scores)<.025 or 'unconfirmed_change' in state['quality_flags']:
            results.append(fallback);continue
        sampled=[ids[i] for i in np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int)]
        observations={};data={};votes=None;visible=None
        for fid in sampled:
            f=measure_features(small(cv2.imread(str(root/frames[fid]['image'])),config),config.board)
            data[fid]=f
            observations[fid]=(np.packbits(f['ink']),np.packbits(f['foreground']))
            votes=f['ink'].astype(np.uint16) if votes is None else votes+f['ink']
            visible=(~f['foreground']).astype(np.uint16) if visible is None else visible+(~f['foreground'])
        support=(votes>=2)&(votes>=.4*visible)
        static=ctx['mask'] if ctx else np.zeros(support.shape,bool)
        eligible=[fid for fid in sampled if (data[fid]['foreground'] & ~static).mean()>.025
                  and (support & data[fid]['foreground'] & ~static).sum()>=20]
        if len(eligible)<4:
            results.append(fallback);continue
        candidates=[copy.deepcopy(metrics[f]) for f in eligible]
        ranking=rank_candidates(candidates,observations,support.shape)
        state['source_frame_ids']=eligible
        state['selected_frame_id']=ranking[0]['source_frame_id']
        state['selection_candidates']=ranking[:5]
        state['selection_reason']='same_ranker_on_actual_occluded_sources_only'
        if args.anchor=='max_occlusion':
            anchor=max(eligible,key=lambda f:float((data[f]['foreground'] & ~static).mean()))
            state['selected_frame_id']=anchor
            state['selection_candidates']=[next(x for x in ranking if x['source_frame_id']==anchor)]+ranking[:5]
            state['selection_reason']='max_observed_occlusion_anchor_for_capability_test'
        r=reconstruct(root,folder,state,frames,config,context=ctx)
        results.append(r)
        cases.append(dict(state_id=state['id'],start=state['start'],end=state['end'],
            original_selected_frame_id=next(s['selected_frame_id'] for s in original['board_states'] if s['id']==state['id']),
            occluded_source_ids=eligible,excluded_sources=[f for f in ids if f not in eligible],
            ranking=ranking,result=r))
        print(state['id'],state['start'],state['end'],r['status'],r['quality_flags'],flush=True)
    m['reconstructed_boards']=results
    m['selected_frames']=[dict(board_state_id=s['id'],source_frame_id=s['selected_frame_id'],reason=s['selection_reason']) for s in m['board_states']]
    m['occlusion_challenge']=dict(uses_ground_truth=False,anchor=args.anchor,max_cases=args.max_cases,policy='Source-only states with >=12 observations and foreground range >=.025; sampled sources require dynamic foreground >2.5% and >=20 hidden supported ink pixels; >=4 sources. No clean source pixels used for restoration. Anchor stress test is distinct from native A selection.',cases=cases)
    write(root/'manifest.json',m)
    write(root/'challenge.json',m['occlusion_challenge'])


if __name__=='__main__':main()
