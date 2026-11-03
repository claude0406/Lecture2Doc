"""Accumulated board pages. GT is never consumed; output pixels are source pixels."""
import cv2
import numpy as np
from .vision import small, measure_features
from .storage import write, key
import os

PAGE_VERSION = '2026-11-02.pages-4'


def observations(root, folder, frames, config, progress=None):
    """Checkpoint packed measurements, independently of page decision thresholds."""
    previous = folder
    folder = root / 'cache' / 'page-observations' / key([frames,config.board,config.analysis_width,config.roi], 'page-measurements-1')
    folder.mkdir(parents=True, exist_ok=True)
    for old in previous.glob('features-*.npz'):
        dest=folder/old.name
        if not dest.exists():
            os.link(old,dest)
    chunks = []
    for start in range(0, len(frames), 128):
        path = folder / f'features-{start:06d}.npz'
        if not path.exists():
            inks, masks, sharp = [], [], []
            for f in frames[start:start+128]:
                image = cv2.imread(str(root / f['image']))
                if image is None:
                    raise ValueError(f"Unreadable source {f['id']}")
                m = measure_features(small(image, config), config.board)
                inks.append(np.packbits(m['ink']))
                masks.append(np.packbits(m['foreground']))
                sharp.append(m['ink_sharpness'])
                shape = m['ink'].shape
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, ink=inks, mask=masks, sharp=sharp, shape=shape)
            os.replace(temp,path)
        with np.load(path) as data:
            chunks.append({k: data[k].copy() for k in data.files})
        if progress:
            progress(f"완성 판서 관측 {min(start+128,len(frames))}/{len(frames)}")
    if not chunks:
        return None
    return dict(ink=np.concatenate([c['ink'] for c in chunks]),
                mask=np.concatenate([c['mask'] for c in chunks]),
                sharp=np.concatenate([c['sharp'] for c in chunks]), shape=tuple(chunks[0]['shape']))


def group_pages(frames, data, config, duration, metrics=(), diagnostic=None):
    """Accumulate additions; split only on sustained, visible loss of old writing.

    Missing pixels underneath foreground never vote for erasure. Deletion sweeps
    have a settling period so each partly wiped frame cannot create a page.
    """
    if not frames:
        return dict(pages=[], events=[])
    data=dict(data,ink=data['ink'].copy(),mask=data['mask'].copy())
    alignment_events=[]; aligned_ids=set()
    shape = tuple(map(int,data['shape'])); n = int(np.prod(shape)); kernel = np.ones((3,3),np.uint8)
    def unpack(kind,i):
        return np.unpackbits(data[kind][i],count=n).reshape(shape).astype(bool)
    def near(a):
        return cv2.dilate(a.astype('uint8'),kernel).astype(bool)
    metric_by_id={m['id']:m for m in metrics}
    for i,f in enumerate(frames):
        metric=metric_by_id.get(f['id'],{}); camera=metric.get('camera',{})
        matrix=camera.get('transform')
        if metric.get('event')=='camera_motion' and camera.get('verified') and matrix:
            mat=np.asarray(matrix,dtype='float32')
            if (abs(mat[0,2])<shape[1]*.05 and abs(mat[1,2])<shape[0]*.05
                and abs(camera.get('scale',1)-1)<.03 and abs(camera.get('rotation_degrees',0))<2):
                for j in range(i):
                    for kind in ('ink','mask'):
                        warped=cv2.warpAffine(unpack(kind,j).astype('uint8'),mat,(shape[1],shape[0]),
                            flags=cv2.INTER_NEAREST,borderValue=1 if kind=='mask' else 0)
                        data[kind][j]=np.packbits(warped)
                aligned_ids.add(f['id'])
                alignment_events.append(dict(time=f['time'],kind='analysis_camera_alignment',transform=matrix,
                                             output_pixels_unchanged=True))
    # Static frame/rail components at the image edge are not inscriptions.
    # Derive exclusions from source observations, never from annotation masks.
    sample=np.linspace(0,len(frames)-1,min(64,len(frames)),dtype=int)
    frequency=sum(unpack('ink',int(j)).astype('float32') for j in sample)/len(sample)
    common=(frequency>.5).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(common,8)
    fixtures=np.zeros(shape,bool);h,w=shape
    for label in range(1,count):
        x,y,cw,ch,area=stats[label]
        rail=(cw>w*.5 and ch<h*.15 and (y<h*.18 or y+ch>h*.82))
        if rail:
            fixtures |= labels==label
    # A pair of long, persistent opposite rails provides a source-only board
    # rectangle. Without that evidence, retain the configured ROI in full.
    edge_support=frequency>.2
    rows=edge_support.sum(1);cols=edge_support.sum(0)
    top=np.flatnonzero((np.arange(h)<h*.2)&(rows>w*.45))
    bottom=np.flatnonzero((np.arange(h)>h*.8)&(rows>w*.45))
    left=np.flatnonzero((np.arange(w)<w*.2)&(cols>h*.45))
    right=np.flatnonzero((np.arange(w)>w*.8)&(cols>h*.45))
    bounds=[0,0,w,h]
    if len(top) and len(bottom):
        bounds[1]=int(top.max()+3);bounds[3]=int(bottom.min()-2)
    if len(left) and len(right):
        bounds[0]=int(left.max()+3);bounds[2]=int(right.min()-2)
    xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    fixtures |= (xx<bounds[0])|(xx>=bounds[2])|(yy<bounds[1])|(yy>=bounds[3])
    foreground_frequency=sum(unpack('mask',int(j)).astype('float32') for j in sample)/len(sample)
    persistent_unknown=(foreground_frequency>.5)&~fixtures
    unknown_area=float(persistent_unknown.mean())
    fixtures=near(fixtures | persistent_unknown)
    raw_unpack=unpack
    def unpack(kind,i):
        value=raw_unpack(kind,i)
        mask=cv2.dilate(raw_unpack('mask',i).astype('uint8'),np.ones((7,7),np.uint8)).astype(bool)
        return (value & ~mask & ~fixtures) if kind=='ink' else (mask & ~fixtures)
    times = np.array([f['time'] for f in frames])
    # Counts express elapsed observation time, including with non-default sampling.
    confirm = max(2, int(np.ceil(3.0/config.sample_seconds)))
    absence = max(3, int(np.ceil(2.0/config.sample_seconds)))
    stable = np.zeros(shape,np.uint16); missing=stable.copy()
    reference=np.zeros(shape,bool)
    history={}
    last_page_target=None
    pages=[]; events=list(alignment_events); start=0; erasing=False; last_loss=-1e9; scene=0; pending_loss=0; pending_target=None; pending_onset=0; pending_started=-1e9; pending_removed=np.zeros(shape,bool)
    by_metric={m['id']:m for m in metrics}
    def emit(stop,target,reason):
        nonlocal start,last_page_target
        if stop <= start or int(target.sum()) < 100:
            return
        candidates=[]
        for j in range(start,stop):
            ink,mask=unpack('ink',j),unpack('mask',j)
            readable=float((target & near(ink) & ~mask).sum()/max(1,target.sum()))
            hidden=float((target & mask).sum()/max(1,target.sum()))
            extra=float((ink & ~near(target)).sum()/max(1,target.sum()))
            score=readable-.6*extra-.15*hidden-.03*float(mask.mean())+.002*min(1,float(data['sharp'][j])/80)
            candidates.append(dict(source_frame_id=frames[j]['id'],time=float(times[j]),score=score,
                                   readable_fraction=readable,hidden_fraction=hidden,extra_fraction=extra,
                                   foreground_fraction=float(mask.mean())))
        candidates.sort(key=lambda c:(c['score'],c['time']),reverse=True)
        best=candidates[0]
        if reason=='confirmed_erasure_or_revision' and last_page_target is not None:
            novel=int((target & ~near(last_page_target)).sum())
            if novel<max(60,int(target.sum()*.05)) and best['readable_fraction']<.97:
                events.append(dict(time=float(times[stop]),kind='erasure_continuation_not_a_completed_page',
                    novel_pixels=novel,best_readable_fraction=best['readable_fraction']))
                return False
        flags=[]
        if best['readable_fraction']<.97:
            flags.append('complete_original_not_observed')
        if unknown_area>.03:
            flags.append('persistent_foreground_unobserved')
        if best['foreground_fraction']>.03:
            flags.append('foreground_present_original_used')
        if best['extra_fraction']>.05:
            flags.append('terminal_content_uncertain')
        page=dict(id=f'p{len(pages):05d}',scene=scene,start=float(times[start]),
                  end=float(times[stop]) if stop<len(frames) else duration,
                  source_frame_ids=[f['id'] for f in frames[start:stop]],
                  selected_frame_id=best['source_frame_id'],selection_reason='terminal_readable_writing',
                  selection_candidates=candidates[:8],boundary_reason=reason,
                  quality_flags=flags,transcript_ids=[],target_ink_pixels=int(target.sum()),
                  original_complete_estimate=not flags)
        pages.append(page)
        last_page_target=target.copy()
        if diagnostic:
            write(diagnostic/(page['id']+'.candidates.json'),dict(page=page,candidates=sorted(candidates,key=lambda c:c['time'])))
            cv2.imwrite(str(diagnostic/(page['id']+'.target.png')),target.astype('uint8')*255)
        return True
    for i,f in enumerate(frames):
        ink,mask=unpack('ink',i),unpack('mask',i)
        metric=by_metric.get(f['id'],{})
        # A verified geometric discontinuity invalidates pixel correspondence.
        discontinuity=metric.get('event') in ('scene_cut','camera_motion') and f['id'] not in aligned_ids
        if discontinuity and i>start:
            emit(i,reference.copy(),'camera_or_scene_change')
            events.append(dict(time=float(times[i]),kind='camera_or_scene_change'))
            reference[:]=False;stable[:]=0;missing[:]=0;start=i;scene+=1;erasing=False;pending_loss=0;pending_target=None;pending_removed[:]=False;history.clear();last_page_target=None
        visible=~mask
        stable=np.where(ink, np.minimum(stable+1,60000),np.where(mask,stable,0)).astype('uint16')
        confirmed=stable>=confirm
        old=reference.copy()
        history[i]=old
        history.pop(i-absence-2,None)
        missing=np.where(visible & reference & ~near(ink),np.minimum(missing+1,60000),0).astype('uint16')
        removed=missing>=absence
        loss=int(removed.sum())
        threshold=max(60,int(old.sum()*.08))
        if times[i]-pending_started>4:
            pending_loss=0;pending_target=None;pending_removed[:]=False
        if loss:
            if pending_target is None:
                pending_started=float(times[i]);pending_onset=max(start+1,i-absence+1)
                pending_target=history.get(pending_onset,old).copy()
            pending_removed |= removed
            pending_loss=int(pending_removed.sum())
        if erasing and pending_loss>=threshold*.25:
            last_loss=float(times[i])
        if pending_loss>=threshold:
            onset=pending_onset
            if not erasing:
                emitted=emit(onset,pending_target,'confirmed_erasure_or_revision')
                if emitted:
                    start=onset
                erasing=True
            events.append(dict(time=float(times[i]),onset=float(times[onset]),kind='visible_writing_loss',
                               removed_pixels=pending_loss,reference_pixels=int(old.sum())))
            last_loss=float(times[i]);pending_loss=0;pending_target=None;pending_removed[:]=False
        reference |= confirmed
        reference[removed]=False
        missing[removed]=0
        if erasing and times[i]-last_loss>=4:
            erasing=False
        # Do not erase a small correction silently: log losses below page threshold.
        if loss and loss<threshold and loss>=20:
            events.append(dict(time=float(times[i]),kind='minor_loss',removed_pixels=loss,threshold=threshold))
    emit(len(frames),reference,'end_of_video')
    if not pages:
        # A short or heavily occluded clip must still retain an observed source.
        counts=[int(unpack('ink',j).sum()) for j in range(len(frames))]
        if max(counts,default=0)>=100:
            j=max(range(len(frames)),key=lambda j:(counts[j],times[j]))
            emit(len(frames),unpack('ink',j),'insufficient_stable_observations')
            pages[-1]['quality_flags'].append('insufficient_stable_observations')
            pages[-1]['original_complete_estimate']=False
    return dict(pages=pages,events=events,policy=dict(version=PAGE_VERSION,
        addition_confirmation_seconds=3,visible_absence_seconds=2,erasure_fraction=.08,
        minimum_erasure_pixels=60,erasure_settle_seconds=4,uses_ground_truth=False,
        output_pixels='selected_original_only',source_inferred_board_bounds=bounds,persistent_unknown_fraction=unknown_area))


def build_pages(root,folder,frames,config,duration,metrics,progress=None):
    data=observations(root,folder,frames,config,progress)
    result=group_pages(frames,data,config,duration,metrics,folder) if data else dict(pages=[],events=[])
    for page in result['pages']:
        page['terminal_ink_mask']=(folder/(page['id']+'.target.png')).relative_to(root).as_posix()
    return result
