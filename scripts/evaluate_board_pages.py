"""Page-output diagnosis with unchanged AccessMath pixel scoring functions."""
import argparse,sys,html
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read,write,digest
from accessmath_evaluate import annotations,predicted_ink,score


def evaluate(record,baseline,revised,gt,out):
    out.mkdir(parents=True,exist_ok=True)
    m=read(record/'manifest.json');old=read(baseline/'manifest.json');micro=read(revised/'manifest.json')
    keys,_=annotations(gt);frames={f['id']:f for f in m['source_frames']}
    rows=[]
    for k in keys:
        if k['time']>=m['source']['duration']:continue
        page=next(p for p in m['board_pages'] if p['start']<=k['time']<p['end'])
        state=next(s for s in old['board_states'] if s['start']<=k['time']<s['end'])
        ms=next(s for s in micro['board_states'] if s['start']<=k['time']<s['end'])
        of={f['id']:f for f in old['source_frames']};mf={f['id']:f for f in micro['source_frames']}
        b=next(r for r in old['reconstructed_boards'] if r['board_state_id']==state['id'])
        nearest=min(m['source_frames'],key=lambda f:abs(f['time']-k['time']))
        f=frames[page['selected_frame_id']]
        images=dict(GT=k['image'],nearest=cv2.imread(str(record/nearest['image'])),
            baseline_A=cv2.imread(str(baseline/of[state['selected_frame_id']]['image'])),
            baseline_B=cv2.imread(str(baseline/b['image'])),
            previous_micro_A=cv2.imread(str(revised/mf[ms['selected_frame_id']]['image'])),
            page_A=cv2.imread(str(record/f['image'])))
        page_b=next((r for r in m.get('page_reconstructions',[]) if r['board_page_id']==page['id']),None)
        if page_b:images['page_B']=cv2.imread(str(record/page_b['image']))
        else:images['page_B_original_fallback']=images['page_A']
        row=dict(gt_time=k['time'],gt_index=k['index'],page_id=page['id'],selected_time=f['time'],
            baseline_selected_time=of[state['selected_frame_id']]['time'],metrics={})
        for name,image in images.items():
            row['metrics'][name]={str(t):score(predicted_ink(image,k['ink'].shape,t),k['ink'],k['valid']) for t in (10,20,30)}
        sheet=Image.new('RGB',(1440,600),'white');draw=ImageDraw.Draw(sheet)
        for j,name in enumerate(('GT','nearest','baseline_A','baseline_B','page_A','page_B') if page_b else ('GT','nearest','baseline_A','baseline_B','previous_micro_A','page_A')):
            x=(j%3)*480;y=(j//3)*300
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(images[name],(480,270)),cv2.COLOR_BGR2RGB)),(x,y+25))
            draw.text((x+6,y+5),f"{name} F1={row['metrics'][name]['10']['f1']*100:.2f}%",fill='black')
        sheet.save(out/f"gt-{k['index']}.jpg",quality=94)
        rows.append(row)
    labels=rows[0]['metrics'].keys()
    aggregate={str(t):{label:{metric:float(np.mean([r['metrics'][label][str(t)][metric] for r in rows])) for metric in ('recall','precision','f1')} for label in labels} for t in (10,20,30)}
    write(out/'evaluation.json',dict(page_restoration_results=m.get('page_reconstructions',[]),record=str(record),baseline=str(baseline),revised_micro=str(revised),
        pages=len(m['board_pages']),baseline_microstates=len(old['board_states']),previous_microstates=len(micro['board_states']),
        scoring_unchanged=True,scoring_source_sha256=digest(Path('scripts/accessmath_evaluate.py')),
        prediction_policy='Completed page containing GT timestamp, no GT candidate optimization. B uses page restoration output if present; otherwise it is explicit original fallback.',
        caveat='A completed page can contain writing added after an official snapshot. This is a page-output diagnosis, not the official unique-CC metric.',
        aggregate=aggregate,rows=rows))
    doc=['<!doctype html><meta charset="utf-8"><title>누적 판서 A 평가</title><style>body{font:16px sans-serif;max-width:1500px;margin:32px auto;background:#f4f6f2;color:#18332d}img{max-width:100%}section{background:white;padding:20px;margin:20px 0}a{color:#167358}.sources{display:flex;flex-wrap:wrap;gap:8px}.sources img{width:220px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style><h1>누적 판서 A — 원본과 선정 근거</h1>',f"<p>기존 {len(old['board_states'])}개 중간 상태 → 누적 판서 {len(m['board_pages'])}개. A는 원본 픽셀만 사용합니다. 페이지 B가 있으면 실제 결과를 비교하며, 원본 유지와 합성은 JSON의 상태·사유로 구분합니다.</p>",'<table><tr><th>출력</th><th>Recall</th><th>Precision</th><th>F1</th></tr>']
    for label,values in aggregate['10'].items():
        doc.append('<tr><td>'+label+'</td>'+''.join(f'<td>{values[x]*100:.2f}%</td>' for x in ('recall','precision','f1'))+'</tr>')
    doc.append('</table><p>기존 이진화·회수율·정밀도·F1 계산을 그대로 사용했습니다. 완성 판서는 GT 스냅샷 이후의 추가 필기를 포함할 수 있습니다. GT는 평가에만 사용했습니다.</p>')
    import os
    def href(path):return html.escape(os.path.relpath(path,out),quote=True)
    for row in rows:
        doc.append(f"<section><h2>GT {row['gt_time']:.3f}초 · {row['page_id']} · 선택 {row['selected_time']:.3f}초</h2><img src=\"gt-{row['gt_index']}.jpg\"></section>")
    for p in m['board_pages']:
        f=frames[p['selected_frame_id']];best=p['selection_candidates'][0]
        doc.append(f'<section><h2>{p["id"]} · {p["start"]:.2f}–{p["end"]:.2f}초 · 대표 {f["time"]:.3f}초</h2><p>확인된 최종 판서 노출 추정 {best["readable_fraction"]*100:.1f}% · 경계 {p["boundary_reason"]} · 품질 표시 {html.escape(str(p["quality_flags"]))}</p><a href="{href(record/f["image"])}"><img width="960" src="{href(record/f["image"])}"></a><details><summary>대표 후보·원본 관측 ({len(p["source_frame_ids"])}개 보존)</summary><div class="sources">')
        ids={c['source_frame_id'] for c in p['selection_candidates']}
        ids.update(p['source_frame_ids'][::max(1,len(p['source_frame_ids'])//12)])
        for fid in p['source_frame_ids']:
            if fid not in ids:continue
            sf=frames[fid];doc.append(f'<a href="{href(record/sf["image"])}"><img loading="lazy" src="{href(record/sf["image"])}"><br>{sf["time"]:.3f}초</a>')
        diag=m.get('page_diagnostics')
        if diag:doc.append(f'</div><p><a href="{href(record/diag/(p["id"]+".candidates.json"))}">모든 후보의 선정 점수 JSON</a></p></details></section>')
        else:doc.append('</div></details></section>')
    (out/'index.html').write_text('\n'.join(doc))
    thumbs=[]
    for page in m['board_pages']:
        f=frames[page['selected_frame_id']]
        image=cv2.resize(cv2.imread(str(record/f['image'])),(480,270))
        cv2.putText(image,f"{page['id']} {page['start']:.0f}-{page['end']:.0f} pick {f['time']:.0f}",(8,22),0,.5,(0,0,255),2)
        thumbs.append(image)
    while len(thumbs)%3:thumbs.append(np.full_like(thumbs[0],245))
    cv2.imwrite(str(record/'pages-contact.jpg'),np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0,len(thumbs),3)]))
    print(record,len(m['board_pages']),aggregate['10'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--revised',type=Path,required=True);p.add_argument('--gt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.record,a.baseline,a.revised,a.gt,a.out)
