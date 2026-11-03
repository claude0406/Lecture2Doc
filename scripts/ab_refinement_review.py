"""Build a review of frozen A/B results. This script does not select outputs."""
import html,json,os,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from lecture2doc.storage import read,write,digest
from lecture2doc.page_restore import corrected_patch
from accessmath_evaluate import predicted_ink,score
root=Path('experiments/ab-refinement');out=root/'review';out.mkdir(exist_ok=True)
def href(path):return html.escape(os.path.relpath(path,out),quote=True)
def img(path):return f'<a href="{href(path)}"><img loading="lazy" src="{href(path)}"></a>'
def table(values):
 return '<table><tr><th>출력</th><th>Recall</th><th>Precision</th><th>F1</th></tr>'+''.join('<tr><td>'+html.escape(label)+'</td>'+''.join(f'<td>{v[k]*100:.2f}%</td>' for k in ['recall','precision','f1'])+'</tr>' for label,v in values.items())+'</table>'
parts=['<!doctype html><meta charset="utf-8"><title>A/B 실제 영상 재평가</title><style>body{font:16px/1.6 sans-serif;background:#f5f7f2;color:#18332d;max-width:1500px;margin:30px auto;padding:12px}section{background:white;padding:24px;margin:24px 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}a{color:#146548}.warning{background:#fff1d6;padding:16px}</style><h1>A/B 실제 영상 재평가</h1><p>2026-11-02 · A pages-17 / B page-restoration-19. 기존 실험은 보존했습니다. C는 시작하지 않았습니다.</p><p class="warning">일반 실행과 가림 구간 능력 시험은 별개입니다. 일반 실행 26장은 모두 원본 유지입니다. B 가림 시험의 참고 이미지는 공식 GT가 아닙니다. 합성 경계·미관측 영역·분할 실패가 남아 있습니다.</p>']
summary={}
for name,title in [('lecture01','Lecture 01 part 1 · training'),('lecture02-testing','Lecture 02 · official testing, 개발 중 진단에 이미 사용'),('lecture06-additional','Lecture 06 part 1 · 추가 training 영상')]:
 record=root/'accepted'/name;ev=root/'accepted-evaluation'/name;data=read(ev/'evaluation.json');m=read(record/'manifest.json');summary[name]=data['aggregate']['10']
 parts.append(f'<section><h2>{title}</h2><p>누적 판서 {len(m["board_pages"])}개 · GT {len(data["rows"])}개 · B 원본 유지 {sum(r["status"]=="original" for r in m["page_reconstructions"])}개.</p>'+table(data['aggregate']['10'])+f'<p><a href="{href(ev/"index.html")}">GT·가장 가까운 관측·이전 A·새 A/B 전체 비교</a> · <a href="{href(record/"manifest.json")}">판서·후보·복원 이유 JSON</a></p>')
 fs={f['id']:f for f in m['source_frames']};thumbs=[]
 for page in m['board_pages']:
  f=fs[page['selected_frame_id']];im=cv2.resize(cv2.imread(str(record/f['image'])),(480,270));cv2.putText(im,f"{page['id']} {page['start']:.0f}-{page['end']:.0f} / {f['time']:.1f}s",(8,20),0,.45,(0,0,255),1);thumbs.append(im)
 while len(thumbs)%3:thumbs.append(np.full_like(thumbs[0],245))
 contact=out/(name+'-pages.jpg');cv2.imwrite(str(contact),np.vstack([np.hstack(thumbs[j:j+3]) for j in range(0,len(thumbs),3)]));parts.append(img(contact))
 for row in data['rows']:
  if name=='lecture02-testing' and row['gt_index']==13238:parts.append('<h3>작은 행렬 수정 보존: 이전 A 96.73% → 현재 A '+f"{row['metrics']['A']['10']['f1']*100:.2f}%</h3>"+img(ev/row['comparison']))
  if name=='lecture01' and row['gt_index']==21894:parts.append('<h3>남은 약화 사례: 이전 누적 A와 비교</h3>'+img(ev/row['comparison']))
 if name=='lecture06-additional':parts.append('<p class="warning">이전 435–443초 삭제 중간 상태는 18개의 고립된 1~3픽셀 잡음을 새 필기로 오인했습니다. pages-17은 이런 잡음만으로 새 완성 판서를 만들지 않아 7장→6장으로 줄었습니다. 억제한 삭제 구간은 앞 판서에 귀속시켜 다음 주제의 시각·출처와 섞이지 않게 합니다. 최초 pages-14는 06 GT 열람 전에 고정했고 그 출력은 verified/lecture06-unseen에 보존했습니다. 현재 pages-17은 01의 후보 동률 보완 및 06의 삭제 잡음 보완 뒤 재실행한 결과이고, 06은 소스와 GT를 이미 본 추가 진단입니다. 따라서 최종 A/B의 엄격한 미사용 testing 검증으로 주장하지 않습니다. 영상 1440×1080(SAR 4:3)은 평가에서 기존 함수대로 GT 크기에 맞춰 비교하며, 반해상도 정합 오차는 JSON에 기록했습니다.</p>')
 parts.append('</section>')
# Specific user-reported failure: keep the nearest clean source as a candidate,
# not as an algorithm input or an oracle selection rule.
r=root/'accepted/lecture01';m=read(r/'manifest.json');fs={f['id']:f for f in m['source_frames']};p=next(p for p in m['board_pages'] if p['start']<=607.674<p['end']);diag=read(r/m['page_diagnostics']/(p['id']+'.candidates.json'));oldroot=Path('experiments/accessmath/lecture01_part1');old=read(oldroot/'manifest.json');oldp=next(s for s in old['board_states'] if s['start']<=607.674<s['end']);oldf=next(f for f in old['source_frames'] if f['id']==oldp['selected_frame_id']);nearest=min(fs.values(),key=lambda f:abs(f['time']-607.674));sel=fs[p['selected_frame_id']]
focus=[min(diag['candidates'],key=lambda c:abs(c['time']-t)) for t in [607.607,616.1155,sel['time']]]
write(out/'607-candidates.json',dict(uses_gt_for_selection=False,page_id=p['id'],page_start=p['start'],page_end=p['end'],selected_time=sel['time'],selection_tolerance_ink_pixels=p['selection_tolerance_ink_pixels'],focus_candidates=focus,all_candidates_path=str(r/m['page_diagnostics']/(p['id']+'.candidates.json')),legacy_selected_time=oldf['time'],legacy_selection_reason=oldp['selection_reason']))
sheet=Image.new('RGB',(1440,300),'white');draw=ImageDraw.Draw(sheet)
for j,(path,label) in enumerate([(oldroot/oldf['image'],f"Legacy A {oldf['time']:.3f}s"),(r/nearest['image'],f"Clean observation {nearest['time']:.3f}s"),(r/sel['image'],f"Current accumulated A {sel['time']:.3f}s")]):
 im=cv2.imread(str(path));sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,28));draw.text((j*480+5,5),label,fill='black')
sheet.save(out/'607-comparison.jpg',quality=95)
parts.append('<section><h2>607초 대표 선정</h2>'+img(out/'607-comparison.jpg')+f'<p>607.607초 원본을 버리지 않고 후보로 보존합니다. GT 시각과의 거리를 점수로 사용하지 않습니다. 삭제 직전 누적 판서 p00001의 대표는 {sel["time"]:.3f}초입니다. 점수는 말기 필기 노출 − 0.6×다른 필기 − 0.15×필기 가림이며, 5 분석 픽셀 이내 동등 후보는 전경이 적은 쪽을 우선합니다.</p><p><a href="607-candidates.json">세 후보 점수·선정 근거</a> · <a href="{href(r/m["page_diagnostics"]/(p["id"]+".candidates.json"))}">전체 후보 JSON</a> · <a href="{href(Path("experiments/accessmath-improvement/diagnosis/candidates-607.html"))}">이전 33개 후보의 코드 경로 진단</a></p></section>')
allcases=[]
for name in ['intervals01-verified','intervals02-verified']:
 d=read(root/name/'results.json')
 parts.append('<section><h2>실제 연속 가림 구간 · '+name+'</h2><p>정상 A와 별개인 능력 시험입니다. 모든 0.5초 관측에 가림이 있는 연속 구간에서 A/말기 근거/B를 구간 내부 관측만으로 만들었습니다. 밖의 깨끗한 원본은 완료 후 평가에만 사용했습니다. 공식 GT 점수와 합산하지 않습니다.</p>')
 for case in d['results']:
  allcases.append(case);r=case['result'];q=case['proposal'];parts.append(f'<h3>{q["start"]:.3f}–{q["end"]:.3f}초 · {r["status"]} · {r["reason"]}</h3>'+table(case['metrics'])+img(root/name/case['comparison'])+f'<p>추가 노출 추정 {r.get("estimated_added_ink_pixels",0)} 분석 픽셀. 이 수치는 읽을 수 있는 글자 수가 아닙니다.</p>')
  for image in r.get('diagnostics',[]):parts.append(img(root/name/image))
  if r.get('provenance'):parts.append(f'<a href="{href(root/name/r["provenance"])}">영역별 원본·대조 관측·밝기 보정 출처</a>')
 parts.append('</section>')
mean={l:{k:float(np.mean([c['metrics'][l][k] for c in allcases])) for k in ['recall','precision','f1']} for l in ['A','B']}
parts.append('<section><h2>실제 가림 시험 집계 · 공식 GT 아님</h2>'+table(mean)+'<p>8건 중 7건 합성/부분 복원, 1건 원본 유지. 구간 선택은 복원 전에 고정했으며 판서 주제·가림 종류 전체를 대표하는 무작위 시험은 아닙니다. 몸통 일부와 패치 경계가 남아 점수 상승만으로 완성품이라고 판단하지 않습니다.</p><p><a href="../provenance-verification.json">채택 7건 출처의 픽셀 단위 재생 검증</a> · <a href="../intervals02-final/dense-audit.json">두 구간의 원본 모든 프레임 180+120장 가림 검사</a></p><p>밀집 검사에서도 최소 가림은 각각 확인 필기 397/16970, 532/9276 픽셀이었습니다. 휴리스틱 가림 추정의 검사이며 모든 강의에 깨끗한 전체 장면이 없다는 증명은 아닙니다.</p></section>')
bad=read(root/'b-paired-lecture01/challenge.json');badcase=next(c for c in bad['cases'] if c['state_id']=='b00054')
parts.append('<section><h2>실제로 악화된 개발 후보 · 채택 금지 회귀 사례</h2>'+table(badcase['metrics'])+img(root/'b-paired-lecture01/b00054.jpg')+'<p>개발 중 B의 원본 교체가 A에만 보이던 필기를 잃어 F1 98.44→93.39%로 악화됐습니다. 이는 현재 채택 결과가 아닙니다. 현재는 기존 A 가시 필기 소실이 5픽셀을 넘는 원본 교체를 금지합니다. 실제 잔상/패치 경계 실패는 위 591초 부분 복원과 2158초 원본 유지에서도 확인할 수 있습니다.</p></section>')
# Synthetic result remains a separate controlled experiment.
syn=root/'current/synthetic';sm=read(syn/'manifest.json');sp=sm['board_pages'][0];sr=sm['page_reconstructions'][0];sf={f['id']:f for f in sm['source_frames']};truth=cv2.imread('experiments/page-restoration/synthetic-complementary/truth.png');A=cv2.imread(str(syn/sf[sp['selected_frame_id']]['image']));B=cv2.imread(str(syn/sr['image']));valid=np.ones(truth.shape[:2],bool);metrics={name:score(predicted_ink(im,valid.shape,10),predicted_ink(truth,valid.shape,10),valid) for name,im in [('A',A),('B',B)]};write(out/'synthetic-evaluation.json',dict(synthetic=True,actual_video_claim=False,metrics=metrics,result=sr))
sheet=Image.new('RGB',(1440,300),'white');draw=ImageDraw.Draw(sheet)
for j,(im,label) in enumerate([(A,'SYNTHETIC A'),(B,'SYNTHETIC B'),(truth,'Synthetic reference')]):sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB)),(j*480,28));draw.text((j*480+5,5),label,fill='black')
sheet.save(out/'synthetic-comparison.jpg',quality=95);parts.append('<section><h2>통제된 합성 영상 시험</h2>'+table(metrics)+img(out/'synthetic-comparison.jpg')+'<p>실제 영상 성능과 구분합니다. 원본에 없는 필기를 생성하지 않습니다.</p></section>')
write(out/'summary.json',dict(A_version='2026-11-02.pages-17',B_version='2026-11-02.page-restoration-19',official=summary,real_intervals_reference_only=mean,synthetic=metrics,software_tests_log='../validation/A17-full-tests.log',code_sha256={str(p):digest(p) for p in [Path('lecture2doc/pages.py'),Path('lecture2doc/page_restore.py'),Path('lecture2doc/vision.py')]}))
if (out/'viewer-A17-B19.png').exists():parts.append('<section><h2>최종 뷰어 확인</h2>'+img(out/'viewer-A17-B19.png')+'</section>')
(out/'index.html').write_text('\n'.join(parts));print(summary,mean,metrics,flush=True)
