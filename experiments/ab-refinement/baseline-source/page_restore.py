"""Restore an accumulated page from observations matching its terminal writing.

No GT, generated background, or pixels from another writing state are used.
Unknown regions retain the A source. Every copied rectangle has provenance.
"""
from functools import lru_cache
import cv2
import numpy as np
from .vision import small, measure_features, camera_evidence, surface_foreground
from .reconstruct import _inconsistent, scene_context
from .storage import write, read

PAGE_RESTORATION_VERSION = '2026-11-02.page-restoration-11'


def person_mask(image,measured):
    """Extend thick foreground to connected skin/striped-clothing boundaries.

    Thin writing is not a person: colour components need a thick core and must
    touch the neighbourhood of an already detected foreground component.
    """
    raw=measured['foreground']
    _,surface=surface_foreground(image,measured['board'])
    delta=image.astype(float)-surface
    red=(delta[:,:,2]-delta[:,:,0]>12)&(delta[:,:,2]-delta[:,:,1]>4)&(np.max(np.abs(delta),axis=2)>20)
    core=cv2.morphologyEx(red.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    core=cv2.morphologyEx(core,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(core)
    neighbourhood=cv2.dilate(raw.astype('uint8'),np.ones((25,25),np.uint8)).astype(bool)
    keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=20 and np.any(neighbourhood&(labels==i))]
    extension=cv2.dilate(np.isin(labels,keep).astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    return cv2.dilate(raw.astype('uint8'),np.ones((7,7),np.uint8)).astype(bool)|extension


def illumination_fit(base,donor,visible,ink_a,ink_b):
    """Fit only observed blank board, never text or an occluded pixel."""
    h,w=visible.shape
    blank=visible & ~cv2.dilate((ink_a|ink_b).astype('uint8'),np.ones((7,7),np.uint8)).astype(bool)
    yy,xx=np.nonzero(blank)
    if len(xx)<200:return None
    step=max(1,len(xx)//8000);xx,yy=xx[::step],yy[::step]
    x,y=2*xx/w-1,2*yy/h-1
    basis=np.stack((x*0+1,x,y,x*y,x*x,y*y),axis=1)
    difference=base[yy,xx].astype(float)-donor[yy,xx]
    weights=np.ones(len(xx))
    for _ in range(4):
        coefficients=np.linalg.lstsq(basis*weights[:,None],difference*weights[:,None],rcond=None)[0]
        residual=np.max(np.abs(difference-basis@coefficients),axis=1)
        weights=np.minimum(1,5/np.maximum(residual,1))
    error=float(np.percentile(residual,90))
    if error>10:return None
    return dict(coefficients=coefficients.tolist(),fit_p90_error=error,
        coordinate_system='normalized_roi_-1_to_1',rule='add smooth illumination offset fitted to co-visible blank board')


def corrected_patch(source,x,y,width,height,roi_box,correction):
    if correction is None:return source
    X0,Y0,RW,RH=roi_box
    yy,xx=np.mgrid[y:y+height,x:x+width]
    xx,yy=2*(xx-X0)/RW-1,2*(yy-Y0)/RH-1
    basis=np.stack((xx*0+1,xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)
    offset=basis@np.asarray(correction['coefficients'])
    if np.max(np.abs(offset))>40:return None
    return np.clip(np.rint(source.astype(float)+offset),0,255).astype('uint8')


def restore_page(root, folder, page, frames, config, policy=None, states=(), context=None):
    selected=frames[page['selected_frame_id']]
    result=dict(id='pr'+page['id'][1:],board_page_id=page['id'],status='original',
        image=selected['image'],source_frame_ids=[selected['id']],provenance=None,
        quality_flags=[],reason='clear_original_sufficient',coverage=None,diagnostics=[],
        estimated_added_ink_pixels=0, lost_visible_ink_pixels=0)
    if not config.reconstruction:
        result['reason']='reconstruction_disabled';return result
    base_full=cv2.imread(str(root/selected['image']))
    base=small(base_full,config);initial=measure_features(base,config.board)
    h,w=base.shape[:2];kernel=np.ones((3,3),np.uint8)
    near=lambda a: cv2.dilate(a.astype('uint8'),kernel).astype(bool)
    bounds=(policy or {}).get('source_inferred_board_bounds',[0,0,w,h])
    board=np.zeros((h,w),bool);x,y,x2,y2=bounds;board[y:y2,x:x2]=True
    # Only the board interior matters; a lecturer outside it is not a defect.
    initial['restoration_foreground']=person_mask(base,initial)
    mask=lambda m:m['restoration_foreground']
    if context is not None:
        board &= ~context['mask']
        result['mask_context']={k:v for k,v in context.items() if k!='mask'}
    target=mask(initial)&board
    result['occluded_board_fraction']=float(target.sum()/max(1,board.sum()))
    ranked=next((r for r in page.get('selection_candidates',[]) if r['source_frame_id']==selected['id']),{})
    if target.sum()<max(12,board.sum()*.003):return result
    if (ranked.get('readable_fraction',0)>=.99 and ranked.get('hidden_fraction',1)<.01
        and not any(f in page['quality_flags'] for f in ('complete_original_not_observed','persistent_foreground_unobserved'))):
        return result
    expected=cv2.imread(str(root/page['terminal_ink_mask']),cv2.IMREAD_GRAYSCALE)>0 if page.get('terminal_ink_mask') else None
    if expected is not None and expected.shape!=(h,w):
        result['reason']='terminal_target_geometry_mismatch';return result
    expected_near=near(expected) if expected is not None else None
    barriers=[]
    ids=page['source_frame_ids']
    terminal=next((s for s in states if selected['id'] in s['source_frame_ids']),None)
    if terminal is None:
        result['reason']='terminal_state_unavailable';return result
    if terminal:
        if any(f in terminal['quality_flags'] for f in ('unconfirmed_change','camera_motion_reconstruction_disabled')):
            result['reason']='unsafe_terminal_state';return result
        if expected is None:
            allowed=set(terminal['source_frame_ids']);ids=[fid for fid in ids if fid in allowed]
        else:
            # Additive/motion micro-boundaries may be crossed only with the
            # terminal ink target; confirmed deletion boundaries are barriers.
            barriers=[s['end'] for s in states if s['boundary_reason']=='erasure_or_revision']
            ids=[fid for fid in ids if not any(min(frames[fid]['time'],selected['time'])<t<=max(frames[fid]['time'],selected['time']) for t in barriers)]
    if expected is None:
        ids=[fid for fid in ids if abs(frames[fid]['time']-selected['time'])<=12]
    if len(ids)>64:
        nearby=sorted(ids,key=lambda fid:abs(frames[fid]['time']-selected['time']))[:32]
        diverse=[ids[i] for i in np.linspace(0,len(ids)-1,32,dtype=int)]
        alternatives=[r['source_frame_id'] for r in page.get('selection_candidates',[]) if r['source_frame_id'] in ids]
        ids=list(dict.fromkeys(nearby+diverse+alternatives))
    ids=[selected['id']]+sorted(set(ids)-{selected['id']},key=lambda fid:(abs(frames[fid]['time']-selected['time']),-frames[fid]['time']))
    result['source_policy']=dict(rule='same accumulated page with terminal-target checks, no erasure boundary, mutually compatible writing. Without terminal target: same microstate and within 12 seconds.',
        terminal_target_used=expected is not None,terminal_target=page.get('terminal_ink_mask'),erasure_barriers=barriers,
        microstate_id=terminal['id'] if terminal else None,candidate_frame_ids=ids,accepted_frame_ids=[],rejected=[])
    if len(ids)<3:
        result['reason']='insufficient_same_state_observations';return result
    observations=[]
    for fid in ids:
        im=base if fid==selected['id'] else small(cv2.imread(str(root/frames[fid]['image'])),config)
        m=initial if fid==selected['id'] else measure_features(im,config.board)
        if 'restoration_foreground' not in m:m['restoration_foreground']=person_mask(im,m)
        fg=mask(m);ink=m['ink'] & ~fg
        shared=board & ~fg & ~mask(initial)
        if fid!=selected['id']:
            moving,evidence=camera_evidence(base,im,mask(initial),fg)
            phase=evidence.get('phase',[0,0])
            reason=None
            transform=evidence.get('transform')
            translated=transform is not None and np.linalg.norm(np.asarray(transform)[:,2])>.75
            if moving or (evidence.get('verified') and translated):
                reason='changed_camera_geometry'
            elif not evidence.get('verified') and max(abs(phase[0]),abs(phase[1]))>.75:
                # A moving person can dominate whole-image phase correlation.
                # Verify near-identity alignment on co-visible board pixels.
                co_ink=(initial['ink'] & ink & shared).sum()
                try:
                    corr,warp=cv2.findTransformECC(cv2.cvtColor(base,cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),np.eye(2,3,dtype='float32'),
                        cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-4),
                        inputMask=shared.astype('uint8')*255,gaussFiltSize=3)
                    if co_ink<40 or corr<.97 or np.linalg.norm(warp[:,2])>.75:
                        reason='unverified_camera_geometry'
                except cv2.error:
                    reason='unverified_camera_geometry'
            if reason is None and _inconsistent(initial['ink'],ink,shared,initial['contrast'],m['contrast']):
                reason='different_visible_writing'
            if reason is None and expected is not None:
                required=expected & board & ~fg
                missing=required & ~near(ink)
                # The accumulated target records confirmed positive strokes;
                # its unmarked pixels are not certified blank background.
                if missing.sum()>max(12,required.sum()*.01):
                    reason='different_terminal_writing'
            if reason is None:
                for prior in observations[1:]:
                    both=board & ~fg & ~prior['mask']
                    if _inconsistent(prior['ink'],ink,both,prior['contrast'],m['contrast']):
                        reason='donor_writing_conflict';break
            if reason:
                result['source_policy']['rejected'].append(dict(source_frame_id=fid,reason=reason));continue
        observations.append(dict(id=fid,ink=ink,mask=fg,image=im,contrast=m['contrast']))
    result['source_policy']['accepted_frame_ids']=[o['id'] for o in observations]
    if len(observations)<3:
        result['reason']='insufficient_compatible_observations';return result
    for o in observations:
        o['clearance']=cv2.distanceTransform((~o['mask']).astype('uint8'),cv2.DIST_L2,3)
        o['near_ink']=near(o['ink'])
        o['illumination']=illumination_fit(base,o['image'],board & ~mask(initial) & ~o['mask'],initial['ink'],o['ink'])
    ink_votes=sum(o['ink'].astype('uint16') for o in observations)
    visible_votes=sum((~o['mask']).astype('uint16') for o in observations)
    supported=(ink_votes>=2)&(ink_votes>=visible_votes*.6)&board
    if expected is not None:supported &= expected_near
    hidden=supported&target
    result['observed_hidden_ink_pixels']=int(hidden.sum())
    if hidden.sum()<max(8,supported.sum()*.005):
        result['reason']='no_repeatedly_observed_hidden_writing';return result
    # Prefer one genuinely clear compatible original to any composite.
    base_visible=initial['ink'] & ~mask(initial) & board
    reference=(expected & board) if expected is not None else supported
    anchor_readable=float((reference & near(initial['ink']) & ~mask(initial)).sum()/max(1,reference.sum()))
    clear_options=[]
    for o in observations[1:]:
        readable=float((reference & o['near_ink'] & ~o['mask']).sum()/max(1,reference.sum()))
        if readable>=.99 and readable>anchor_readable+.01:
            clear_options.append((readable,-float((o['mask']&board).sum()),o))
    if clear_options:
        o=max(clear_options,key=lambda c:c[:2])[2]
        result.update(image=frames[o['id']]['image'],source_frame_ids=[o['id']],
            reason='clear_compatible_original_preferred',estimated_added_ink_pixels=int((hidden & ~o['mask'] & o['near_ink']).sum()))
        return result
    output=base_full.copy();H,W=base_full.shape[:2]
    rx,ry,rw,rh=config.roi
    X0,Y0=int(rx*W),int(ry*H);RW,RH=int((rx+rw)*W)-X0,int((ry+rh)*H)-Y0
    target_native=np.zeros((H,W),bool)
    target_native[Y0:Y0+RH,X0:X0+RW]=cv2.resize(target.astype('uint8'),(RW,RH),interpolation=cv2.INTER_NEAREST).astype(bool)
    patches=[];resolved=np.zeros((h,w),bool);used={selected['id']}
    @lru_cache(maxsize=4)
    def original(fid):return cv2.imread(str(root/frames[fid]['image']))
    def tile(x,y,x2,y2):
        region=target[y:y2,x:x2]
        if not region.any():return
        choices=[]
        for o in observations[1:]:
            if not o['mask'][y:y2,x:x2].any():
                if expected is not None and (o['ink'][y:y2,x:x2] & target[y:y2,x:x2] & ~expected_near[y:y2,x:x2]).sum()>2:
                    continue
                # Distance from foreground is safer than maximising texture.
                clearance=float(o['clearance'][y:y2,x:x2].min())
                choices.append((clearance,o))
        choices.sort(key=lambda c:c[0],reverse=True)
        pair=None
        for ai,(clearance,a) in enumerate(choices):
            for _,b in choices[ai+1:]:
                if abs(frames[a['id']]['time']-frames[b['id']]['time'])<1:continue
                local=np.ones((y2-y,x2-x),bool)
                if _inconsistent(a['ink'][y:y2,x:x2],b['ink'][y:y2,x:x2],local,
                    a['contrast'][y:y2,x:x2],b['contrast'][y:y2,x:x2]):continue
                da=a['image'][y:y2,x:x2].astype(float);db=b['image'][y:y2,x:x2].astype(float)
                if np.percentile(np.abs(da-db),90)>18:continue
                # Both donors must contain all supported strokes in this patch.
                required=supported[y:y2,x:x2]
                if any((required & ~o['near_ink'][y:y2,x:x2]).sum()>max(2,required.sum()*.02) for o in (a,b)):continue
                pair=(a,b);break
            if pair:break
        if pair is None:
            if max(x2-x,y2-y)<=4:return
            xm,ym=(x+x2)//2,(y+y2)//2
            for a,b,c,d in ((x,y,xm,ym),(xm,y,x2,ym),(x,ym,xm,y2),(xm,ym,x2,y2)):
                if c>a and d>b:tile(a,b,c,d)
            return
        a,b=pair
        X,Y,X2,Y2=X0+round(x*RW/w),Y0+round(y*RH/h),X0+round(x2*RW/w),Y0+round(y2*RH/h)
        # Never overwrite already visible source pixels outside the occluder.
        # Merge equal horizontal runs into provenance rectangles.
        local_mask=target_native[Y:Y2,X:X2]
        patch_source=original(a['id'])[Y:Y2,X:X2]
        output[Y:Y2,X:X2][local_mask]=patch_source[local_mask]
        active={}
        def add_rect(key,y_start,y_end):
            left,right=key
            patches.append(dict(x=X+left,y=Y+y_start,width=right-left,height=y_end-y_start,
                source_frame_id=a['id'],witness_frame_id=b['id'],
                source_box=[X+left,Y+y_start,X+right,Y+y_end]))
        for row in range(local_mask.shape[0]+1):
            runs=[]
            if row<local_mask.shape[0]:
                edges=np.diff(np.r_[False,local_mask[row],False].astype('int8'))
                runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
                runs=[(int(left),int(right)) for left,right in runs]
            for key in list(active):
                if key not in runs:add_rect(key,active.pop(key),row)
            for key in runs:active.setdefault(key,row)
        resolved[y:y2,x:x2] |= region
        used.add(a['id'])
    for y in range(0,h,config.tile_size):
        for x in range(0,w,config.tile_size):tile(x,y,min(w,x+config.tile_size),min(h,y+config.tile_size))
    added=int((hidden & resolved).sum())
    result['attempted_added_ink_pixels']=added
    result['coverage']=float((target&resolved).sum()/max(1,target.sum()))
    if added<max(8,supported.sum()*.005):
        result['reason']='no_safe_hidden_writing_gain';return result
    def reject_proposal(reason):
        path=folder/(page['id']+'.rejected.png');cv2.imwrite(str(path),output)
        result['rejected_proposal']=path.relative_to(root).as_posix()
        result['reason']=reason
        write(folder/(page['id']+'.rejected.provenance.json'),dict(patches=patches,base_source_frame_id=selected['id'],rejected_reason=reason))
        return result
    # Keep only donor rectangles that survive the post-composition foreground
    # check. This can salvage safe writing without accepting a bad whole mask.
    pruned=0;photometric_attempted=False
    for iteration in range(4):
        after_image=small(output,config)
        after=measure_features(after_image,config.board)
        after['restoration_foreground']=person_mask(after_image,after)
        residual=mask(after)&board
        unsafe=residual & resolved
        if unsafe.sum()<=max(8,int((target&resolved).sum()*.02)):
            break
        if not photometric_attempted:
            photometric_attempted=True
            for patch in patches:
                source=next(o for o in observations if o['id']==patch['source_frame_id'])
                correction=source['illumination']
                X,Y,pw,ph=(patch[k] for k in ('x','y','width','height'))
                adjusted=corrected_patch(original(source['id'])[Y:Y+ph,X:X+pw],X,Y,pw,ph,(X0,Y0,RW,RH),correction)
                if adjusted is not None and correction is not None:
                    output[Y:Y+ph,X:X+pw]=adjusted
                    patch['photometric_correction']=correction
            continue
        if iteration==3:return reject_proposal('foreground_residual_original_used')
        unsafe_native=np.zeros((H,W),bool)
        unsafe_native[Y0:Y0+RH,X0:X0+RW]=cv2.resize(near(unsafe).astype('uint8'),(RW,RH),interpolation=cv2.INTER_NEAREST).astype(bool)
        kept=[];copied=np.zeros((H,W),bool)
        for patch in patches:
            X,Y,pw,ph=(patch[k] for k in ('x','y','width','height'))
            if unsafe_native[Y:Y+ph,X:X+pw].any():
                output[Y:Y+ph,X:X+pw]=base_full[Y:Y+ph,X:X+pw];pruned+=1
            else:
                kept.append(patch);copied[Y:Y+ph,X:X+pw]=True
        patches=kept
        resolved=cv2.resize(copied[Y0:Y0+RH,X0:X0+RW].astype('float32'),(w,h),interpolation=cv2.INTER_AREA)>.99
        added=int((hidden&resolved).sum())
        if added<max(8,supported.sum()*.005):return reject_proposal('no_safe_hidden_writing_gain')
    result['pruned_rectangles']=pruned
    result['residual_foreground_pixels']=int((residual&resolved).sum())
    result['coverage']=float((target&resolved).sum()/max(1,target.sum()))
    result['feature_lost_visible_ink_pixels']=int((base_visible & ~near(after['ink'])).sum())
    if not np.array_equal(output[~target_native],base_full[~target_native]):
        return reject_proposal('visible_writing_loss_original_used')
    result['lost_visible_ink_pixels']=0
    recovered=(hidden&resolved&near(after['ink'])).sum()
    if recovered<added*.95:
        return reject_proposal('hidden_writing_not_preserved')
    path=folder/(page['id']+'.png');cv2.imwrite(str(path),output)
    provenance=folder/(page['id']+'.provenance.json')
    write(provenance,dict(coordinate_system='full_frame_pixels',width=W,height=H,roi=config.roi,
        base_source_frame_id=selected['id'],patches=patches,source_policy=result['source_policy'],
        inpainting=False,spatial_interpolation=False,roi_pixel_box=[X0,Y0,RW,RH],photometric_adjustment=any(p.get('photometric_correction') for p in patches),unresolved_pixels='retain selected original'))
    remaining=target&~resolved
    overlay=cv2.resize(output,(w,h),interpolation=cv2.INTER_AREA)
    overlay[resolved&target]=(overlay[resolved&target]*.35+np.array([40,190,40])*.65).astype('uint8')
    overlay[remaining]=(overlay[remaining]*.35+np.array([30,70,220])*.65).astype('uint8')
    diagnostic=folder/(page['id']+'.regions.png');cv2.imwrite(str(diagnostic),overlay)
    result.update(status='composited' if remaining.sum()<=max(8,target.sum()*.005) else 'partial',
        image=path.relative_to(root).as_posix(),source_frame_ids=sorted({selected['id']}|{p['source_frame_id'] for p in patches}),
        provenance=provenance.relative_to(root).as_posix(),diagnostics=[diagnostic.relative_to(root).as_posix()],
        reason='observed_hidden_writing_recovered',estimated_added_ink_pixels=added,quality_flags=['heuristic_restoration_requires_review'])
    result['verification_frame_ids']=sorted({p['witness_frame_id'] for p in patches})
    result['photometric_adjusted']=any(p.get('photometric_correction') for p in patches)
    if result['photometric_adjusted']:result['quality_flags'].append('source_illumination_adjusted')
    if remaining.any():result['quality_flags'].append('unobserved_regions_original_retained')
    return result


def restore_pages(root,folder,pages,frames,config,policy=None,states=(),progress=None):
    by_id={f['id']:f for f in frames};results=[]
    contexts=scene_context(root,folder,states,by_id,config) if config.reconstruction and states else {}
    for i,page in enumerate(pages):
        checkpoint=folder/(page['id']+'.result.json')
        if checkpoint.exists():result=read(checkpoint)
        else:
            try:result=restore_page(root,folder,page,by_id,config,policy,states,contexts.get(page['scene']))
            except Exception as exc:
                result=dict(id='pr'+page['id'][1:],board_page_id=page['id'],status='original',
                    image=by_id[page['selected_frame_id']]['image'],source_frame_ids=[page['selected_frame_id']],
                    provenance=None,quality_flags=['page_restoration_failed'],reason='page_restoration_failed',error=f'{type(exc).__name__}: {exc}',diagnostics=[],coverage=None)
            write(checkpoint,result)
        results.append(result)
        if progress:progress(f'누적 판서 복원 {i+1}/{len(pages)} · {result["status"]} · {result["reason"]}')
    return results
