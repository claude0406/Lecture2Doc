import sys
from pathlib import Path
import cv2
import numpy as np
from lecture2doc.page_restore import restore_page
from lecture2doc.storage import read
from lecture2doc.vision import Config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fixtures import render,board


def setup_sources(root, images):
    frames={}
    for i,im in enumerate(images):
        # Visible fixed landmarks make camera alignment observable even when
        # the two actor positions jointly cover all the central writing.
        im=im.copy()
        for x,y in ((25,55),(1040,55),(1040,660)):
            cv2.putText(im,'B1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(90,90,90),2)
        images[i]=im
        fid=f'f{i:04d}';path=root/(fid+'.png');cv2.imwrite(str(path),im)
        frames[fid]=dict(id=fid,time=i*.5,image=path.name)
    ids=list(frames)
    page=dict(id='p00000',selected_frame_id=ids[0],source_frame_ids=ids,quality_flags=[],selection_candidates=[])
    state=dict(id='b00000',source_frame_ids=ids,quality_flags=[],boundary_reason='end_of_video',end=len(images)*.5)
    folder=root/'b';folder.mkdir()
    return frames,page,state,folder


def test_observed_patches_recover_occluded_writing_with_exact_provenance(tmp_path):
    images=[render('occlusion',i*.5,white=True)[0] for i in range(16)]
    frames,page,state,folder=setup_sources(tmp_path,images)
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status'] in ('composited','partial'),r
    assert r['estimated_added_ink_pixels']>10
    assert r['lost_visible_ink_pixels']<=5
    out=cv2.imread(str(tmp_path/r['image']))
    prov=read(tmp_path/r['provenance'])
    replay=images[0].copy()
    for patch in prov['patches']:
        x,y,w,h=(patch[k] for k in ('x','y','width','height'))
        source=cv2.imread(str(tmp_path/frames[patch['source_frame_id']]['image']))
        assert np.array_equal(out[y:y+h,x:x+w],source[y:y+h,x:x+w])
        replay[y:y+h,x:x+w]=source[y:y+h,x:x+w]
    assert np.array_equal(out,replay)
    assert prov['inpainting'] is False


def test_clean_original_is_preferred_and_never_visible_stays_original(tmp_path):
    images=[render('occlusion',i*.5,white=True)[0] for i in range(8)]+[board(2,white=True)]*4
    frames,page,state,folder=setup_sources(tmp_path,images)
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status']=='original'
    assert r['reason']=='clear_compatible_original_preferred',r
    assert r['provenance'] is None


def test_no_pixels_from_a_different_microstate(tmp_path):
    covered=render('never_visible',0,white=True)[0]
    frames,page,state,folder=setup_sources(tmp_path,[covered]*6+[board(2,white=True)]*6)
    state['source_frame_ids']=list(frames)[:6]
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status']=='original',r
    assert set(r['source_frame_ids'])<=set(state['source_frame_ids'])
    assert set(r['source_policy']['candidate_frame_ids'])<=set(state['source_frame_ids'])


def test_different_visible_writing_is_rejected(tmp_path):
    covered=render('occlusion',0,white=True)[0]
    frames,page,state,folder=setup_sources(tmp_path,[covered]*4+[board(3,white=True)]*4)
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status']=='original'
    assert any(x['reason']=='different_visible_writing' for x in r['source_policy']['rejected'])


def test_illumination_coefficients_are_replayable_and_do_not_move_writing():
    from lecture2doc.page_restore import illumination_fit,corrected_patch
    im=np.full((120,200,3),180,np.uint8)
    cv2.putText(im,'x=2',(60,75),cv2.FONT_HERSHEY_SIMPLEX,1,(30,30,30),2)
    donor=(im.astype('int16')+18).astype('uint8')
    ink=(im[:,:,0]<100)
    fit=illumination_fit(im,donor,np.ones(ink.shape,bool),ink,ink)
    restored=corrected_patch(donor,0,0,200,120,(0,0,200,120),fit)
    assert np.array_equal(restored,im)
    assert fit['fit_p90_error']<1e-5


def test_terminal_target_can_bridge_motion_boundaries_but_not_erasures(tmp_path):
    from lecture2doc.vision import small,measure_features
    images=[render('occlusion',i*.5,white=True)[0] for i in range(16)]
    frames,page,state,folder=setup_sources(tmp_path,images)
    truth=board(2,white=True)
    for x,y in ((25,55),(1040,55),(1040,660)):
        cv2.putText(truth,'B1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(90,90,90),2)
    target=measure_features(small(truth,Config()),'whiteboard')['ink']
    cv2.imwrite(str(tmp_path/'terminal.png'),target.astype('uint8')*255)
    page['terminal_ink_mask']='terminal.png'
    ids=list(frames)
    states=[dict(id='b0',source_frame_ids=ids[:4],quality_flags=[],end=2.,boundary_reason='writing_progress'),
            dict(id='b1',source_frame_ids=ids[4:],quality_flags=[],end=8.,boundary_reason='end_of_video')]
    r=restore_page(tmp_path,folder,page,frames,Config(),states=states)
    assert r['status'] in ('composited','partial'),r
    assert any(fid in ids[4:] for fid in r['source_frame_ids'])
    states[0]['boundary_reason']='erasure_or_revision'
    r=restore_page(tmp_path,folder,page,frames,Config(),states=states)
    assert r['status']=='original'
    assert set(r['source_policy']['candidate_frame_ids'])<=set(ids[:4])


def test_missing_state_evidence_preserves_original(tmp_path):
    frames,page,state,folder=setup_sources(tmp_path,[render('occlusion',i*.5,white=True)[0] for i in range(8)])
    r=restore_page(tmp_path,folder,page,frames,Config())
    assert r['status']=='original'
    assert r['reason']=='terminal_state_unavailable'


def test_temporal_absence_prevents_resurrecting_old_writing(tmp_path):
    from lecture2doc.vision import small,measure_features
    images=[render('occlusion',i*.5,white=True)[0] for i in range(16)]
    frames,page,state,folder=setup_sources(tmp_path,images)
    shape=small(images[0],Config()).shape[:2]
    # All observations predate a later confirmed blank. Even repeated old
    # strokes cannot be certified as current writing in a hidden region.
    np.savez_compressed(tmp_path/'evidence.npz',last_blank_seconds=np.full(shape,100.,np.float32),last_positive_seconds=np.full(shape,101.,np.float32))
    page['terminal_absence_evidence']='evidence.npz'
    target=measure_features(small(board(2,white=True),Config()),'whiteboard')['ink']
    cv2.imwrite(str(tmp_path/'target.png'),target.astype('uint8')*255)
    page['terminal_ink_mask']='target.png'
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status']=='original',r
    assert r['reason']=='no_repeatedly_observed_hidden_writing',r
    assert r['observed_hidden_ink_pixels']==0
    assert any(x['stale_ink_pixels']>0 for x in r['source_policy']['region_exclusions'])


def test_temporal_restoration_replays_source_and_brightness_evidence(tmp_path):
    from lecture2doc.vision import small,measure_features
    from lecture2doc.page_restore import corrected_patch
    images=[render('occlusion',i*.5,white=True)[0] for i in range(16)]
    frames,page,state,folder=setup_sources(tmp_path,images)
    target=measure_features(small(board(2,white=True),Config()),'whiteboard')['ink']
    np.savez_compressed(tmp_path/'evidence.npz',last_blank_seconds=np.full(target.shape,-1.,np.float32),last_positive_seconds=np.full(target.shape,7.5,np.float32))
    cv2.imwrite(str(tmp_path/'target.png'),target.astype('uint8')*255)
    page.update(terminal_absence_evidence='evidence.npz',terminal_ink_mask='target.png')
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status'] in ('partial','composited'),r
    prov=read(tmp_path/r['provenance']);replay=images[0].copy()
    for patch in prov['patches']:
        x,y,w,h=(patch[k] for k in ('x','y','width','height'))
        source=images[list(frames).index(patch['source_frame_id'])][y:y+h,x:x+w]
        replay[y:y+h,x:x+w]=corrected_patch(source,x,y,w,h,prov['roi_pixel_box'],patch.get('photometric_correction'))
    assert np.array_equal(replay,cv2.imread(str(tmp_path/r['image'])))
    assert r['lost_visible_ink_pixels']==0


def test_border_safety_margin_alone_never_triggers_restoration(tmp_path):
    from lecture2doc.vision import small,measure_features
    image=board(2,white=True);image[:,0:70]=(30,30,30)
    frames,page,state,folder=setup_sources(tmp_path,[image.copy() for _ in range(8)])
    measured=measure_features(small(cv2.imread(str(tmp_path/frames[page['selected_frame_id']]['image'])),Config()),'whiteboard')
    # Exclude the actual frame; only its inflated foreground margin overlaps
    # the board. Repeated texture there must not cause a needless composite.
    mask=np.ones(measured['ink'].shape,np.uint8)*255;mask[:,:36]=0
    cv2.imwrite(str(tmp_path/'board.png'),mask)
    np.savez_compressed(tmp_path/'evidence.npz',last_blank_seconds=np.full(mask.shape,-1.,np.float32),last_positive_seconds=np.full(mask.shape,3.5,np.float32))
    page.update(analysis_board_mask='board.png',terminal_absence_evidence='evidence.npz')
    r=restore_page(tmp_path,folder,page,frames,Config(),states=[state])
    assert r['status']=='original',r
    assert r['reason']=='no_substantial_board_occluder',r
    assert r['board_occluder_core_pixels']==0
