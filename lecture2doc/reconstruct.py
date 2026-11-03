"""Conservative source-pixel restoration after readable-original selection."""
from collections import defaultdict
from functools import lru_cache
import cv2
import numpy as np
from .storage import write
from .vision import crop, features, small, measure_features


def scene_context(root, folder, states, frames_by_id, config):
    """Other states supply mask frequency ONLY, never reconstructed writing pixels."""
    groups = defaultdict(list)
    for state in states:
        groups[state['scene']].extend(state['source_frame_ids'])
    output = {}
    for scene, ids in groups.items():
        if len(ids) < 32 or frames_by_id[ids[-1]]['time']-frames_by_id[ids[0]]['time'] < 30:
            continue
        sample = [ids[i] for i in np.linspace(0,len(ids)-1,min(48,len(ids))).astype(int)]
        path = folder/f'static-fixtures-scene-{scene}.png'
        cached=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) if path.exists() else None
        if cached is not None:
            output[scene]=dict(mask=cached>0,image=path.relative_to(root).as_posix(),frame_ids=sample,
                               rule='foreground in >=95% of scene samples; mask only')
            continue
        hits = None
        for fid in sample:
            im = small(cv2.imread(str(root/frames_by_id[fid]['image'])),config)
            _, mask, _ = features(im,config.board)
            hits = mask.astype(np.uint16) if hits is None else hits+mask
        static = hits >= np.ceil(.95*len(sample))
        # Only persistent border fixtures, not a stationary person in the board
        # interior, may be treated as non-board background.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(static.astype(np.uint8))
        keep = []
        h,w = static.shape
        for i in range(1,n):
            x,y,rw,rh,area = stats[i]
            border = x<=w*.03 or y<=h*.05 or x+rw>=w*.97 or y+rh>=h*.95
            if border and (area<h*w*.015 or max(rw/rh,rh/rw)>6):
                keep.append(i)
        static = np.isin(labels,keep).astype(np.uint8)
        static = cv2.dilate(static,np.ones((3,3),np.uint8)).astype(bool)
        cv2.imwrite(str(path),static.astype(np.uint8)*255)
        output[scene] = dict(mask=static,image=path.relative_to(root).as_posix(),frame_ids=sample,
                             rule='foreground in >=95% of scene samples; mask only')
    return output


def _inconsistent(a,b,shared,contrast_a=None,contrast_b=None):
    kernel = np.ones((3,3),np.uint8)
    near_a = cv2.dilate(a.astype(np.uint8),kernel).astype(bool)
    near_b = cv2.dilate(b.astype(np.uint8),kernel).astype(bool)
    changed = shared & ((a & ~near_b) | (b & ~near_a))
    if contrast_a is not None:
        changed &= np.abs(contrast_a.astype(float)-contrast_b)>10
    n, labels = cv2.connectedComponents(cv2.dilate(changed.astype(np.uint8),kernel))
    counts = np.bincount(labels[changed],minlength=n)
    limit = max(12,int(((a | b) & shared).sum()*.001))
    return bool(len(counts)>1 and counts[1:].max(initial=0)>limit)


def reconstruct(root,folder,state,frames_by_id,config,context=None):
    selected = frames_by_id[state['selected_frame_id']]
    base = crop(cv2.imread(str(root/selected['image'])),config.roi).copy()
    result = dict(id='r'+state['id'][1:],board_state_id=state['id'],status='fallback',
                  image=selected['image'],source_frame_ids=[selected['id']],provenance=None,
                  quality_flags=['model_free_masks_require_review'],coverage=None)
    if not config.reconstruction:
        result['quality_flags']=['reconstruction_disabled']; return result
    if any(x in state['quality_flags'] for x in ('unconfirmed_change','camera_motion_reconstruction_disabled')):
        result['quality_flags'].append('unsafe_state_original_used'); return result
    ids = state['source_frame_ids']
    if len(ids)<2:
        result['quality_flags'].append('insufficient_observations'); return result
    candidates = [x['source_frame_id'] for x in state.get('selection_candidates',[])]
    sample = [ids[i] for i in np.linspace(0,len(ids)-1,min(48,len(ids))).astype(int)]
    sample = list(dict.fromkeys([selected['id']]+candidates+sample))
    # A long BoardState may contain a small undetected revision. Sources need
    # both temporal proximity and co-visible writing agreement, not just its ID.
    sample = [fid for fid in sample if abs(frames_by_id[fid]['time']-selected['time'])<=12.]
    initial=measure_features(small(cv2.imread(str(root/selected['image'])),config),config.board)
    static=context['mask'] if context else np.zeros(initial['ink'].shape,bool)
    target=initial['foreground'] & ~static
    if target.sum()<=max(12,target.size*.002):
        result['quality_flags'].append('clear_original_sufficient')
        result['occluded_fraction']=float(target.mean())
        return result
    observations=[]
    for fid in sample:
        im=small(cv2.imread(str(root/frames_by_id[fid]['image'])),config)
        measured=initial if fid==selected['id'] else measure_features(im,config.board)
        observations.append((fid,measured['ink'],measured['foreground'],im,measured['contrast']))
    h,w=observations[0][1].shape
    static=context['mask'] if context else np.zeros((h,w),bool)
    if context:
        result['mask_context']=dict(image=context['image'],frame_ids=context['frame_ids'],
                                   rule=context['rule'],supplies_output_pixels=False)
    selected_ink,selected_mask=observations[0][1:3]
    target=selected_mask & ~static
    result['occluded_fraction']=float(target.mean())
    if target.sum()<=max(12,h*w*.002):
        result['quality_flags'].append('clear_original_sufficient'); return result
    ink_votes=sum(o[1].astype(np.uint16) for o in observations)
    visible_votes=sum((~o[2]).astype(np.uint16) for o in observations)
    supported=(ink_votes>=2) & (ink_votes>=.4*visible_votes)
    hidden=int((supported & target).sum())
    result['observed_hidden_ink_pixels']=hidden
    if hidden<12:
        result['quality_flags'].append('no_observed_hidden_writing'); return result
    seen,remembered=np.zeros((h,w),bool),np.zeros((h,w),bool)
    remembered_contrast=np.zeros((h,w),np.uint8)
    accepted=[]; rejected=[]
    for fid,ink,mask,im,contrast in observations:
        visible=~mask
        if _inconsistent(ink,remembered,seen & visible,contrast,remembered_contrast):
            rejected.append(fid)
            continue
        remembered[visible]=ink[visible]; remembered_contrast[visible]=contrast[visible]; seen|=visible
        accepted.append((fid,ink,mask,im,contrast))
    result['source_policy']=dict(max_time_distance_seconds=12.,rejected_frame_ids=rejected,
                                 accepted_frame_ids=[o[0] for o in accepted])
    observations=accepted
    if len(observations)<3:
        result['quality_flags'].append('inconsistent_visible_writing_original_used'); return result
    ranks={x['source_frame_id']:x for x in state.get('selection_candidates',[])}
    support=ranks.get(selected['id'],{}).get('readable_support_fraction',0.)
    for fid,ink,mask,im,contrast in observations[1:]:
        if ((mask & ~static).sum()<=max(12,h*w*.002) and fid in ranks
                and ranks[fid]['readable_support_fraction']>=support-.005):
            result.update(image=frames_by_id[fid]['image'],source_frame_ids=[fid])
            result['quality_flags'].append('clear_original_preferred'); return result
    available=sum((~o[2]).astype(np.uint16) for o in observations[1:])
    result['coverage']=float((available>=2).mean())
    result['foreground_coverage']=float((available[target]>=2).mean())
    if np.any(target & (available<2)):
        result['quality_flags'].append('unobserved_foreground_original_used'); return result
    output=base.copy(); H,W=output.shape[:2]; tiles=[];used=set()
    @lru_cache(maxsize=3)
    def original(fid):
        return crop(cv2.imread(str(root/frames_by_id[fid]['image'])),config.roi)
    failed=False
    def tile(x,y,x2,y2):
        nonlocal failed
        X,Y,X2,Y2=round(x*W/w),round(y*H/h),round(x2*W/w),round(y2*H/h)
        if not target[y:y2,x:x2].any():
            tiles.append(dict(x=X,y=Y,width=X2-X,height=Y2-Y,source_frame_id=selected['id'],verified=False))
            used.add(selected['id']); return
        choices=[]
        for fid,ink,mask,im,contrast in observations[1:]:
            if not mask[y:y2,x:x2].any():
                choices.append((float(cv2.Laplacian(im[y:y2,x:x2],cv2.CV_32F).var()),fid))
        if len(choices)<2:
            if max(x2-x,y2-y)<=1:
                failed=True; return
            xm,ym=(x+x2)//2,(y+y2)//2
            for a,b,c,d in ((x,y,xm,ym),(xm,y,x2,ym),(x,ym,xm,y2),(xm,ym,x2,y2)):
                if c>a and d>b: tile(a,b,c,d)
            return
        fid=max(choices)[1]
        # The two best donor patches must also agree on the newly exposed ink.
        donors=sorted(choices,reverse=True)[:2]
        oa=next(o for o in observations if o[0]==donors[0][1])
        ob=next(o for o in observations if o[0]==donors[1][1])
        local=np.ones((y2-y,x2-x),bool)
        if _inconsistent(oa[1][y:y2,x:x2],ob[1][y:y2,x:x2],local,
                         oa[4][y:y2,x:x2],ob[4][y:y2,x:x2]):
            failed=True; return
        output[Y:Y2,X:X2]=original(fid)[Y:Y2,X:X2]
        tiles.append(dict(x=X,y=Y,width=X2-X,height=Y2-Y,source_frame_id=fid,verified=True)); used.add(fid)
    for y in range(0,h,config.tile_size):
        for x in range(0,w,config.tile_size):
            tile(x,y,min(w,x+config.tile_size),min(h,y+config.tile_size))
    if failed:
        result['quality_flags'].append('no_safe_source_tiles_original_used'); return result
    after,residual,_=features(cv2.resize(output,(w,h),interpolation=cv2.INTER_AREA),config.board)
    region=cv2.dilate(target.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    remaining=int((residual & region & ~static).sum()); result['residual_foreground_pixels']=remaining
    if remaining>max(8,int(target.sum()*.02)):
        result['quality_flags'].append('foreground_residual_original_used'); return result
    near_after=cv2.dilate(after.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    visible_ink=selected_ink & ~selected_mask
    lost=int((visible_ink & ~near_after).sum()); result['lost_visible_ink_pixels']=lost
    if lost>max(10,int(visible_ink.sum()*.002)):
        result['quality_flags'].append('visible_writing_loss_original_used'); return result
    if np.array_equal(output,base):
        result['quality_flags'].append('no_verified_improvement'); return result
    path=folder/(state['id']+'.png'); cv2.imwrite(str(path),output)
    provenance=path.with_suffix('.json')
    write(provenance,dict(coordinate_system='roi_pixels',roi=config.roi,width=W,height=H,tiles=tiles))
    result.update(status='composited' if result['coverage']==1 else 'partial',image=path.relative_to(root).as_posix(),
                  source_frame_ids=sorted(used),provenance=provenance.relative_to(root).as_posix())
    return result
