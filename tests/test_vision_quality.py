import sys
from pathlib import Path

import cv2
import numpy as np

from lecture2doc.vision import Config, camera_evidence, measure_features, rank_candidates, small
from lecture2doc.reconstruct import _inconsistent

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fixtures import board


def test_dim_whiteboard_polarity_and_connected_writing():
    white=board(2,white=True)
    white=(white.astype(float)*.55).astype(np.uint8)
    m=measure_features(small(white,Config()))
    assert m['board']=='whiteboard' and m['board_confidence']>.12
    # Strongly shaded but unobscured writing must not be classified wholesale as a person.
    assert m['foreground'].mean()<.02
    assert m['ink'].sum()>500
    black=measure_features(small(board(2),Config()))
    assert black['board']=='blackboard' and black['ink'].sum()>500


def test_readable_board_beats_sharp_striped_foreground():
    clean=board(2,white=True)
    h,w=clean.shape[:2]
    illumination=np.linspace(.6,.9,w)[None,:,None]
    clean=(clean*illumination).astype(np.uint8)
    covered=clean.copy()
    cv2.rectangle(covered,(180,120),(500,710),(60,60,150),-1)
    for y in range(150,700,10):
        cv2.line(covered,(185,y),(495,y),(170,170,210),3)
    observations={};metrics=[]
    for i,im in enumerate([clean,clean,covered,covered]):
        m=measure_features(small(im,Config()))
        fid=str(i);observations[fid]=(np.packbits(m['ink']),np.packbits(m['foreground']))
        metrics.append(dict(id=fid,time=i,occlusion=float(m['foreground'].mean()),
                            motion_fraction=0.,ink_sharpness=m['ink_sharpness']))
    ranking=rank_candidates(metrics,observations,m['ink'].shape)
    assert ranking[0]['source_frame_id'] in ('0','1')


def test_background_motion_vs_moving_actor():
    rng=np.random.default_rng(7)
    background=np.full((240,400,3),140,np.uint8)
    for x,y in rng.integers([20,20],[380,220],size=(180,2)):
        cv2.circle(background,(int(x),int(y)),2,(35,35,35),-1)
    a=background.copy();b=background.copy()
    a[70:230,80:180]=rng.integers(0,255,(160,100,3),dtype=np.uint8)
    b[70:230,130:230]=a[70:230,80:180]
    ma=np.zeros((240,400),bool);mb=ma.copy()
    ma[65:235,75:185]=True;mb[65:235,125:235]=True
    moving,info=camera_evidence(a,b,ma,mb)
    assert not moving
    translated=cv2.warpAffine(background,np.float32([[1,0,6],[0,1,3]]),(400,240))
    moving,info=camera_evidence(background,translated)
    assert moving and info['verified']


def test_local_writing_conflicts_not_diluted_by_canvas():
    a=np.zeros((180,320),bool);b=a.copy();b[60,60:100]=True
    shared=np.ones_like(a)
    assert _inconsistent(a,b,shared)
    contrast_a=np.zeros(a.shape,np.uint8);contrast_b=contrast_a.copy();contrast_b[b]=9
    # A weak threshold fluctuation alone is not evidence of a changed inscription.
    assert not _inconsistent(a,b,shared,contrast_a,contrast_b)
    contrast_b[b]=30
    assert _inconsistent(a,b,shared,contrast_a,contrast_b)


def test_navy_clothing_does_not_become_board_illumination():
    image=board(2,white=True)
    cv2.fillPoly(image,[np.array([[180,350],[400,300],[650,710],[110,710]])],(65,48,30))
    for y in range(370,700,14):cv2.line(image,(200,y),(430,y),(130,130,125),3)
    m=measure_features(small(image,Config()),'whiteboard')
    # A dark blue garment occupying the lower board must remain foreground,
    # including its solid lower corner, rather than being fitted as shading.
    assert m['foreground'][350:380,90:120].mean()>.95
    clean=measure_features(small(board(2,white=True),Config()),'whiteboard')['ink']
    visible=clean & ~m['foreground']
    assert visible.sum()>100
    assert (visible & m['ink']).sum()/visible.sum()>.95


def test_foreground_core_is_distinct_from_compositing_safety_margin():
    image=np.full((240,400,3),190,np.uint8)
    image[80:230,150:260]=(30,45,75)
    m=measure_features(image,'whiteboard')
    assert m['foreground_core'][100:200,170:240].all()
    assert (m['foreground'] & ~m['foreground_core']).sum()>500
    assert not np.any(m['foreground_core'] & ~m['foreground'])
