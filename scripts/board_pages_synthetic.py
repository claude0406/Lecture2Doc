"""Synthetic page-stage observations, separate from real-video evaluation.

This deliberately supplies no camera/cut metadata. Full pipeline cut handling
is exercised by tests/test_system.py, rather than this feature-only diagnosis.
"""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from lecture2doc.vision import Config,small,measure_features
from lecture2doc.pages import group_pages
from fixtures import render,SCENARIOS
for name in SCENARIOS:
    inks=[];masks=[];sharp=[];frames=[]
    for i in range(24):
        m=measure_features(small(render(name,i*.5)[0],Config()))
        inks.append(np.packbits(m['ink']));masks.append(np.packbits(m['foreground']));sharp.append(m['ink_sharpness']);frames.append(dict(id=str(i),time=i*.5))
    data=dict(ink=np.array(inks),mask=np.array(masks),sharp=sharp,shape=m['ink'].shape)
    r=group_pages(frames,data,Config(),12)
    print(name,[(p['start'],p['end'],p['selected_frame_id'],p['quality_flags']) for p in r['pages']],flush=True)
