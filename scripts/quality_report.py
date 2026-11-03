"""Build before/after diagnostics and a report without changing the scoring contract."""
import argparse
import copy
import html
import importlib.util
from pathlib import Path
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,digest
from lecture2doc.vision import Config,small,measure_features,rank_candidates


def diagnostics(base,new,output,archive):
    output.mkdir(parents=True,exist_ok=True)
    before,after=read(base/'manifest.json'),read(new/'manifest.json')
    old_state=next(s for s in before['board_states'] if s['start']<=607.674<s['end'])
    new_state=next(s for s in after['board_states'] if s['start']<=607.674<s['end'])
    old_metrics={m['id']:m for m in before['analysis_metrics']}
    new_metrics={m['id']:m for m in after['analysis_metrics']}
    frames={f['id']:f for f in after['source_frames']}
    old_order=sorted(old_state['source_frame_ids'],key=lambda f:(round(old_metrics[f]['occlusion'],3),-old_metrics[f]['sharpness'],-old_metrics[f]['time']))
    old_rank={f:i+1 for i,f in enumerate(old_order)}
    observations={};candidates=[]
    for fid in old_state['source_frame_ids']:
        a=measure_features(small(cv2.imread(str(new/frames[fid]['image'])),Config()))
        observations[fid]=(np.packbits(a['ink']),np.packbits(a['foreground']))
        candidates.append(copy.deepcopy(new_metrics[fid]))
    same_pool=rank_candidates(candidates,observations,a['ink'].shape)
    modern=sorted((m for m in after['analysis_metrics'] if m['id'] in new_state['source_frame_ids']),key=lambda m:-m['selection']['score'])
    payload=dict(legacy_state=old_state,revised_state=new_state,
                 legacy_candidates=[dict(rank=old_rank[f],**old_metrics[f]) for f in old_order],
                 revised_same_legacy_pool=same_pool,revised_actual_pool=modern,
                 score_rule='readable_support_fraction - .8*foreground_fraction - .15*motion_fraction + .005*min(1,ink_sharpness/80)',
                 ground_truth_used_by_ranker=False)
    white=read(base.parent/'lecture01_part1_whiteboard/manifest.json')
    payload['legacy_whiteboard_state']=next(s for s in white['board_states'] if s['start']<=607.674<s['end'])
    payload['revised_states_over_legacy_whiteboard_period']=[s for s in after['board_states'] if s['start']<670.67 and s['end']>606.1055]
    payload['revised_events_600_675']=[x for x in after['analysis_metrics'] if 600<=x['time']<=675 and x['event']!='none']
    write(output/'candidates-607.json',payload)
    spec=importlib.util.spec_from_file_location('archived_rank_vision',archive/'vision.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    ids=[min(frames,key=lambda f:abs(frames[f]['time']-607.607)),old_state['selected_frame_id'],new_state['selected_frame_id']]
    sheet=Image.new('RGB',(1440,900),'#f5f7f2');d=ImageDraw.Draw(sheet)
    for row,fid in enumerate(ids):
        im=small(cv2.imread(str(new/frames[fid]['image'])),Config())
        _,mask_old,_=old.features(im);m=measure_features(im)
        panels=[]
        for mask in [mask_old,m['foreground']]:
            panel=im.copy();panel[mask]=(.4*panel[mask]+[20,20,150]).astype(np.uint8);panels.append(panel)
        for col,panel in enumerate([im]+panels):
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(panel,(480,270)),cv2.COLOR_BGR2RGB)),(col*480,row*300+28))
        d.text((5,row*300+5),f'{frames[fid]["time"]:.4f}s | source / OLD mask / NEW mask | old_rank={old_rank.get(fid,"outside")} new_score={new_metrics[fid]["selection"]["score"]:.5f}',fill='#19312e')
    sheet.save(output/'selection-before-after.jpg',quality=94)
    page='<!doctype html><meta charset="utf-8"><title>607s selection trace</title><style>body{font:14px system-ui;margin:30px;background:#f5f7f2}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}</style><h1>607초 대표 선택 경로</h1><a href="candidates-607.json">모든 후보와 점수 JSON</a><p>같은 기존 33개 후보에 새 순위 규칙만 적용한 결과와 실제 새 상태의 후보를 분리했습니다.</p><img src="selection-before-after.jpg"><h2>기존 후보 집합을 고정한 순위 비교</h2><table><tr><th>시각</th><th>기존 순위</th><th>새 순위</th><th>점수</th><th>드러난 필기 지지율</th><th>가림</th><th>움직임</th></tr>'
    for rank,c in enumerate(same_pool,1):
        page+=f'<tr><td>{c["time"]:.4f}</td><td>{old_rank[c["source_frame_id"]]}</td><td>{rank}</td><td>{c["score"]:.5f}</td><td>{c["readable_support_fraction"]:.4f}</td><td>{c["foreground_fraction"]:.4f}</td><td>{c["motion_fraction"]:.4f}</td></tr>'
    page+='</table><h2>실제 새 상태의 후보 전부</h2><table><tr><th>순위·시각</th><th>점수</th><th>필기 지지율</th><th>가림</th><th>원본</th></tr>'
    for rank,c in enumerate(modern,1):
        sel=c['selection'];url='../v4/auto/'+frames[c['id']]['image']
        page+=f'<tr><td>{rank} · {c["time"]:.4f}</td><td>{sel["score"]:.5f}</td><td>{sel["readable_support_fraction"]:.4f}</td><td>{sel["foreground_fraction"]:.4f}</td><td><a href="{url}"><img src="{url}" width="220" loading="lazy"></a></td></tr>'
    (output/'candidates-607.html').write_text(page+'</table>',encoding='utf-8')


def comparison_images(before_record,after_record,before_report,after_report,gt,output,prefix):
    old,new=read(before_record/'manifest.json'),read(after_record/'manifest.json')
    of={f['id']:f for f in old['source_frames']};nf={f['id']:f for f in new['source_frames']}
    paths=[]
    for a,b in zip(before_report['keyframes'],after_report['keyframes']):
        # Scoring uses original PNGs through the existing evaluator; these are visual panels only.
        gt_im=cv2.imread(str(gt/'keyframes'/f'{a["gt_index"]}.png'))
        nearest=min(new['source_frames'],key=lambda f:abs(f['time']-a['gt_time']))
        ob=next(r for r in old['reconstructed_boards'] if r['board_state_id']==a['state_id'])
        nb=next(r for r in new['reconstructed_boards'] if r['board_state_id']==b['state_id'])
        panels=[gt_im,cv2.imread(str(before_record/of[a['selected_frame_id']]['image'])),cv2.imread(str(after_record/nf[b['selected_frame_id']]['image'])),
                cv2.imread(str(after_record/nearest['image'])),cv2.imread(str(before_record/ob['image'])),cv2.imread(str(after_record/nb['image']))]
        labels=['GT','OLD A','NEW A','Nearest observation','OLD B','NEW B']
        sheet=Image.new('RGB',(1440,600),'#f5f7f2');d=ImageDraw.Draw(sheet)
        for i,(im,label) in enumerate(zip(panels,labels)):
            x,y=(i%3)*480,(i//3)*300
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(x,y+25));d.text((x+5,y+5),label,fill='#19312e')
        path=f'{prefix}-gt-{a["gt_index"]}.jpg';sheet.save(output/path,quality=94);paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('experiments/accessmath-improvement'));args=p.parse_args()
    root=args.root;diag=root/'diagnosis'
    diagnostics(Path('experiments/accessmath/lecture01_part1'),root/'v4/auto',diag,root/'baseline-source')
    groups=[('training','experiments/accessmath/lecture01_part1','v4/auto','experiments/accessmath/evaluation/report.json','v4/evaluation/report.json','AccessMath2015_lecture_01')]
    if (root/'heldout/evaluation-revised/report.json').exists():
        groups.append(('testing',str(root/'heldout/baseline'),'heldout/revised',str(root/'heldout/evaluation-baseline/report.json'),'heldout/evaluation-revised/report.json','AccessMath2015_lecture_02'))
    summary={};images=[]
    for split,old_record,new_record,old_report,new_report,lecture in groups:
        old,new=read(old_report),read(root/new_report)
        summary[split]=dict(old=old['aggregate_macro'],new=new['aggregate_macro'],old_counts=old['counts'],new_counts=new['counts'],keyframes=[])
        for a,b in zip(old['keyframes'],new['keyframes']):
            summary[split]['keyframes'].append(dict(time=a['gt_time'],index=a['gt_index'],
                old_A=a['metrics']['selected_A']['10'],old_B=a['metrics']['result_B']['10'],
                new_A=b['metrics']['selected_A']['10'],new_B=b['metrics']['result_B']['10']))
        gt=Path('datasets/accessmath/annotations/data/annotations')/lecture
        images+=comparison_images(Path(old_record),root/new_record,old,new,gt,root,split)
    write(root/'summary.json',summary)
    # Preserve a real negative B example from the earlier explicit-whiteboard
    # baseline, with its own A/B scores (not a new accepted reconstruction).
    shutil_source=Path('experiments/accessmath/evaluation-whiteboard/gt-18212.jpg')
    if shutil_source.exists():
        (diag/'legacy-b-regression.jpg').write_bytes(shutil_source.read_bytes())
    page='<!doctype html><meta charset="utf-8"><title>AccessMath before / after</title><style>body{font:16px/1.7 system-ui;margin:35px;background:#f5f7f2;color:#19312e}table{border-collapse:collapse}td,th{border:1px solid #ccd5cc;padding:9px}img{max-width:100%}a{color:#176b51}</style><h1>AccessMath 판서 선택·복원 개선</h1><p>원래 평가 함수·임계값·위치 허용을 유지했습니다. 표는 임계값 10, GT 키프레임별 macro 평균이며 공식 unique-CC 벤치마크 점수가 아닙니다.</p><a href="summary.json">모든 설정·수치 JSON</a> · <a href="diagnosis/candidates-607.html">607초 후보·점수 추적</a> · <a href="v4/evaluation/index.html">새 training GT 비교</a> · <a href="heldout/evaluation-revised/index.html">새 testing GT 비교</a><table><tr><th>분할</th><th>방법</th><th>Recall</th><th>Precision</th><th>F1</th></tr>'
    for split,item in summary.items():
        for version,label in [('old','변경 전'),('new','변경 후')]:
            for stage in ['selected_A','result_B']:
                s=item[version]['10'][stage]
                page+=f'<tr><td>{split}</td><td>{label} {stage}</td><td>{s["recall"]:.2%}</td><td>{s["precision"]:.2%}</td><td>{s["f1"]:.2%}</td></tr>'
    page+='</table><p>합성 횟수는 복원 품질 점수가 아닙니다. 새 B가 원본으로 돌아간 경우 A/B 값이 같습니다. 실제 가림 관측만 남긴 별도 스트레스 테스트는 <a href="occlusion-evaluation/index.html">여기</a>에서 보며, 일반 전체 영상 점수와 구분합니다.</p>'
    page+='<p><a href="occlusion-anchor-evaluation/index.html">가림이 가장 큰 실제 원본을 고정한 별도 앵커 테스트</a> · <a href="synthetic/index.html">합성 영상: 충분한 관측이 있을 때의 복원 성공과 정보 부재 실패</a></p><h2>B의 남은 한계와 거부 근거</h2><p>이번 실제 영상에서 새 B는 합성을 승인하지 않았습니다. 따라서 실제 가림 복원 성공 사례가 생겼다고 보고하지 않습니다. 실제 가림 테스트의 빨간 영역은 두 시점 이상에서 드러난 관측이 없는 영역입니다.</p><img src="diagnosis/real-occlusion-unobserved.jpg"><h2>기존 B의 악화 사례 (whiteboard 별도 기준선)</h2><p>GT 607.674초: A F1 62.6427% → B F1 62.6192%. 아래 교수자 잔상 사례는 수정 전 B이며 새 B의 성공 사례가 아닙니다.</p><img src="diagnosis/legacy-b-regression.jpg"><h2>A의 남은 작은 회귀</h2><p>training 347.781초 F1은 97.2519% → 97.2064%로 0.0455%p 하락했습니다. testing의 세부 증감도 summary.json에 모두 남겼습니다.</p>'
    for path in images:
        page+=f'<h2>{html.escape(path)}</h2><img src="{path}" loading="lazy">'
    (root/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
