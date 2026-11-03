import cv2
import numpy as np
from lecture2doc.pages import group_pages
from lecture2doc.vision import Config


def run(inks,masks=None):
    shape=inks[0].shape
    masks=masks or [np.zeros(shape,bool) for _ in inks]
    frames=[dict(id=str(i),time=i*.5) for i in range(len(inks))]
    data=dict(ink=np.array([np.packbits(x) for x in inks]),mask=np.array([np.packbits(x) for x in masks]),sharp=np.ones(len(inks))*40,shape=shape)
    return group_pages(frames,data,Config(),len(inks)*.5)


def writing():
    a=np.zeros((100,200),bool);a[20:22,20:170]=True
    b=a.copy();b[50:52,20:170]=True
    c=b.copy();c[80:82,20:170]=True
    return a,b,c


def test_additions_accumulate_until_complete():
    a,b,c=writing()
    result=run([a]*8+[b]*8+[c]*8)
    assert len(result['pages'])==1
    p=result['pages'][0]
    assert int(p['selected_frame_id'])>=16
    assert p['selection_candidates'][0]['readable_fraction']==1


def test_multiple_completed_pages_not_one_per_video():
    a,b,c=writing()
    result=run([c]*12+[np.roll(b,5,axis=1)]*12+[np.roll(a,10,axis=1)]*12)
    assert len(result['pages'])==3
    assert [p['end'] for p in result['pages']]==[6,12,18]
    assert all(p['selection_candidates'][0]['readable_fraction']>=.99 for p in result['pages'])
    assert [int(p['selected_frame_id']) for p in result['pages']]==[11,23,35]


def test_foreground_cannot_vote_for_deletion_and_clean_original_wins():
    a,b,c=writing();mask=np.zeros_like(c);mask[:,10:110]=True
    hidden=c&~mask
    result=run([c]*8+[hidden]*12+[c]*8,[np.zeros_like(c)]*8+[mask]*12+[np.zeros_like(c)]*8)
    assert len(result['pages'])==1
    assert result['pages'][0]['selection_candidates'][0]['readable_fraction']==1
    assert int(result['pages'][0]['selected_frame_id']) not in range(8,20)


def test_erasing_sweep_does_not_emit_each_partial_wipe():
    a,b,c=writing()
    empty=np.zeros_like(a)
    result=run([c]*12+[b]*6+[a]*6+[empty]*8+[a]*12)
    assert len(result['pages'])==2
    assert result['pages'][0]['target_ink_pixels']==900


def test_small_verified_camera_move_does_not_create_a_page():
    a,b,c=writing()
    shifted=cv2.warpAffine(c.astype('uint8'),np.float32([[1,0,4],[0,1,0]]),(200,100)).astype(bool)
    inks=[c]*10+[shifted]*14
    frames=[dict(id=str(i),time=i*.5) for i in range(len(inks))]
    data=dict(ink=np.array([np.packbits(x) for x in inks]),mask=np.zeros((24,2500),np.uint8),sharp=np.ones(24)*40,shape=c.shape)
    before=data['ink'].copy()
    metrics=[dict(id='10',event='camera_motion',camera=dict(verified=True,transform=[[1,0,4],[0,1,0]],scale=1,rotation_degrees=0))]
    result=group_pages(frames,data,Config(),12,metrics)
    assert len(result['pages'])==1
    assert result['events'][0]['kind']=='analysis_camera_alignment'
    assert np.array_equal(before,data['ink'])


def test_real_video_pipeline_accumulates_progress_and_retains_erasures(tmp_path):
    from lecture2doc.pipeline import process
    # Three separate finished boards, each held long enough to be observed.
    images=[]
    for topic,count in enumerate((3,2,1)):
        im=np.full((360,640,3),220,np.uint8)
        for j in range(count):
            cv2.putText(im,f'y = {j+1+topic*3}x + 2',(80,100+j*80),cv2.FONT_HERSHEY_SIMPLEX,1,(30,30,30),2)
        images.append(im)
    video=tmp_path/'three-boards.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),4,(640,360))
    for image in images:
        for _ in range(32):writer.write(image)
    writer.release()
    m=process(video,tmp_path/'record',Config(reconstruction=False))
    assert len(m['board_pages'])==3
    ids=[fid for p in m['board_pages'] for fid in p['source_frame_ids']]
    assert ids==[f['id'] for f in m['source_frames']]
    assert all(p['selected_frame_id'] in p['source_frame_ids'] for p in m['board_pages'])


def test_alternating_occlusion_keeps_observed_writing_without_inventing_clean_frame():
    a,b,c=writing()
    left=np.zeros_like(c);left[:,10:100]=True
    right=np.zeros_like(c);right[:,100:190]=True
    masks=([left]*4+[right]*4)*3
    result=run([c&~m for m in masks],masks)
    assert len(result['pages'])==1
    assert 'complete_original_not_observed' in result['pages'][0]['quality_flags']
    assert result['pages'][0]['target_ink_pixels']>=800


def test_short_clip_keeps_an_uncertain_source_instead_of_empty_output():
    a,b,c=writing()
    result=run([c]*4)
    assert len(result['pages'])==1
    assert result['pages'][0]['original_complete_estimate'] is False
    assert 'insufficient_stable_observations' in result['pages'][0]['quality_flags']


def test_small_revision_after_a_stable_board_is_not_diluted_by_total_ink():
    a=np.zeros((360,640),bool)
    for y in range(30,310,25):a[y:y+2,40:350]=True
    a[180:185,450:454]=True
    b=a.copy();b[180:185,450:454]=False;b[180:185,460:464]=True
    result=run([a]*24+[b]*24)
    assert len(result['pages'])==2
    assert result['pages'][0]['end']==12
    assert int(result['pages'][0]['selected_frame_id'])<24


def test_a_correction_while_still_writing_does_not_emit_the_draft():
    a=np.zeros((360,640),bool)
    for y in range(30,310,25):a[y:y+2,40:350]=True
    draft=a.copy();draft[180:185,450:454]=True
    final=a.copy();final[180:185,460:466]=True
    result=run([a]*6+[draft]*8+[final]*20)
    assert len(result['pages'])==1
    assert int(result['pages'][0]['selected_frame_id'])>=14
    assert any(e['kind']=='in_progress_correction' for e in result['events'])


def test_thin_persistent_writing_near_the_border_is_not_a_rail():
    a=np.zeros((100,200),bool);a[95:97,20:170]=True
    result=run([a]*20)
    assert len(result['pages'])==1
    assert result['pages'][0]['target_ink_pixels']>=290


def test_a_late_observed_symbol_is_kept_even_when_an_earlier_frame_is_sharper():
    base=np.zeros((360,640),bool)
    for y in range(30,310,25):base[y:y+2,40:350]=True
    final=base.copy();final[180:182,450:454]=True
    inks=[base]*24+[final]*4
    frames=[dict(id=str(i),time=i*.5) for i in range(len(inks))]
    data=dict(ink=np.array([np.packbits(x) for x in inks]),mask=np.zeros((28,28800),np.uint8),
              sharp=np.array([80]*24+[0]*4),shape=base.shape)
    result=group_pages(frames,data,Config(),14)
    assert len(result['pages'])==1
    assert int(result['pages'][0]['selected_frame_id'])>=24
    assert result['pages'][0]['provisional_ink_pixels']>=8
    assert 'provisional_terminal_writing' in result['pages'][0]['quality_flags']


def test_scattered_threshold_flecks_do_not_make_partial_erasure_a_new_page():
    a,b,c=writing();empty=np.zeros_like(a)
    flecks=empty.copy()
    for x,y in [(10,10),(40,30),(80,40),(120,60),(160,70),(185,90)]:flecks[y,x]=True
    result=run([c]*12+[b|flecks]*12+[a|flecks]*12+[empty]*12+[np.roll(c,10,axis=1)]*12)
    assert len(result['pages'])==2,result['pages']
    assert any(e['kind']=='erasure_continuation_not_a_completed_page' and e.get('raw_novel_pixels',0)>e['novel_pixels'] for e in result['events'])
    assert result['pages'][0]['end']>=12
    assert result['pages'][0]['end']==result['pages'][1]['start']
    assert [fid for p in result['pages'] for fid in p['source_frame_ids']]==[str(i) for i in range(60)]
