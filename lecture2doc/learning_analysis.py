"""Evidence-based lecture importance and board/speech alignment.

This first version uses deterministic text and timeline signals only. It does not
claim semantic understanding: every score is accompanied by inspectable evidence.
"""
from __future__ import annotations

import hashlib
import re

from .storage import stage

LEARNING_ANALYSIS_VERSION = "2026-11-09.learning-importance-1"

_TOKEN = re.compile(
    r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?|[∫∑∏√∂∆∇∞≤≥≠≈→←↔⊂∈]+"
)
_KOREAN_PARTICLES = (
    "으로부터", "에게서", "이라고", "이라는", "이라면", "으로", "에서",
    "에게", "까지", "부터", "처럼", "보다", "만큼", "이고", "이며",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "만",
)
_STOPWORDS = {
    "그리고", "그러나", "그래서", "이것", "저것", "그것", "있다", "한다",
    "하는", "되는", "같은", "통해", "대한", "위해", "때문", "때문에",
    "이제", "여기", "다음", "경우", "말씀", "이렇게", "그런", "어떤",
    "무엇", "정도", "부분", "내용", "설명", "강의", "오늘", "여러분",
    "이거", "저거", "거죠", "겁니다", "합니다", "됩니다", "입니다",
}
_IMPORTANCE_CUES = (
    ("시험·출제 언급", re.compile(r"시험|출제|기출|시험에")),
    ("중요하다는 강조", re.compile(r"중요|핵심|핵심적")),
    ("암기·기억 권고", re.compile(r"기억|외우|암기|잊지")),
    ("주의·오류 경고", re.compile(r"주의|조심|틀리|실수|헷갈")),
    ("정의·원리 설명", re.compile(r"정의|원리|정리하면|공식")),
    ("유도·증명 과정", re.compile(r"유도|증명|도출")),
    ("대표 문제·예제", re.compile(r"예제|대표 문제|연습문제|문제 풀")),
)


def _normal(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣∫∑∏√∂∆∇∞≤≥≠≈→←↔⊂∈]+", "", str(value).lower())


def _tokens(value: str) -> list[str]:
    output = []
    for token in _TOKEN.findall(str(value)):
        token = token.strip().lower()
        if re.fullmatch(r"[가-힣]+", token):
            for particle in _KOREAN_PARTICLES:
                if token.endswith(particle) and len(token) - len(particle) >= 2:
                    token = token[:-len(particle)]
                    break
        if token in _STOPWORDS:
            continue
        if token.isascii() and token.isalpha() and len(token) < 2:
            continue
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        output.append(token)
    return output


def _page_text(ocr_page: dict | None) -> tuple[str, list[dict]]:
    if not ocr_page:
        return "", []
    all_regions = ocr_page.get("regions") or []
    # OCR marks scores below its review threshold as uncertain. Keep those
    # regions out of concept matching so a likely misread does not become
    # a supposedly important lecture concept.
    regions = [region for region in all_regions if not region.get("review", False)]
    text = "\n".join(str(region.get("text", "")) for region in regions if region.get("text"))
    if not all_regions:
        text = str(ocr_page.get("text", ""))
    return text, regions


def _cue_counts(segments: list[dict]) -> tuple[int, list[str]]:
    count = 0
    reasons = []
    for label, pattern in _IMPORTANCE_CUES:
        hits = sum(len(pattern.findall(str(segment.get("text", "")))) for segment in segments)
        if hits:
            count += hits
            reasons.append(f"{label} {hits}회 감지")
    return count, reasons


def _matches(term: str, text: str) -> bool:
    needle = _normal(term)
    raw_text = str(text).lower()
    if not needle or not raw_text.strip():
        return False
    if needle.isascii() and needle.isalpha():
        return bool(re.search(
            r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])",
            raw_text,
        ))
    return needle in _normal(raw_text)


def _extract_terms(text: str, include_pairs: bool = False) -> list[str]:
    words = _tokens(text)
    found = list(words)
    if include_pairs:
        for left, right in zip(words, words[1:]):
            pair = f"{left} {right}"
            if len(_normal(pair)) >= 4:
                found.append(pair)
    # Keep first spelling and order while removing duplicate normalized forms.
    unique = {}
    for term in found:
        key = _normal(term)
        if len(key) >= 2:
            unique.setdefault(key, term)
    return list(unique.values())


def _score_page(segments: list[dict], terms: list[str], ocr_regions: list[dict], has_ocr: bool) -> tuple[int | None, dict, list[str]]:
    cue_count, reasons = _cue_counts(segments)
    if not segments:
        return None, {
            "emphasis": None, "repetition": None,
            "board_speech_alignment": None, "board_concepts": len(terms),
        }, ["음성 전사가 없어 교수자 설명의 중요도를 산정하지 않았습니다"]
    term_counts = [
        sum(1 for segment in segments if _matches(term, segment.get("text", "")))
        for term in terms
    ]
    matched_segments = [
        segment for segment in segments
        if any(_matches(term, segment.get("text", "")) for term in terms)
    ]
    match_count = sum(count > 0 for count in term_counts)
    repeated_term_mentions = sum(max(0, count - 1) for count in term_counts)
    emphasis = min(1.0, cue_count / 2.0)
    repetition = min(1.0, repeated_term_mentions / 3.0)
    alignment = match_count / max(1, len(terms)) if terms else 0.0
    board_signal = 1.0 if has_ocr and terms else 0.0

    weights = []
    if segments:
        weights.extend(((0.40, emphasis), (0.30, repetition)))
    if has_ocr and segments:
        weights.append((0.22, alignment))
    if has_ocr:
        weights.append((0.08, board_signal))
    if not weights:
        return None, {
            "emphasis": None, "repetition": None,
            "board_speech_alignment": None, "board_concepts": len(terms),
        }, ["음성 전사와 OCR 근거가 부족해 중요도를 산정하지 못했습니다"]

    score = round(100 * sum(weight * value for weight, value in weights) /
                  sum(weight for weight, _ in weights))
    if repeated_term_mentions:
        reasons.append(f"관련 개념이 음성에서 {repeated_term_mentions}회 반복 언급")
    if has_ocr and match_count:
        reasons.append(f"칠판 개념 {match_count}개가 음성 설명과 연결")
    if has_ocr and terms:
        reasons.append(f"필기에서 관련 개념 {len(terms)}개 확인")
    if not reasons:
        reasons.append("강의 구간과 필기 자료의 확인 가능한 신호를 종합")
    return score, {
        "emphasis": round(emphasis, 3) if segments else None,
        "repetition": round(repetition, 3) if segments else None,
        "board_speech_alignment": round(alignment, 3) if has_ocr and segments else None,
        "board_concepts": len(terms),
        "emphasis_cue_count": cue_count,
        "matched_speech_segments": len(matched_segments),
        "repeated_term_mentions": repeated_term_mentions,
    }, reasons


def analyze_learning(manifest: dict) -> dict:
    """Build page importance, cross-modal links, and concept appearance times."""
    transcript = manifest.get("transcript") or {}
    segments = sorted(transcript.get("segments") or [], key=lambda item: float(item.get("start", 0)))
    pages = manifest.get("board_pages") or manifest.get("board_states") or []
    ocr_pages = {
        item.get("board_page_id"): item
        for item in (manifest.get("ocr") or {}).get("pages", [])
        if item.get("board_page_id")
    }

    if not pages and not segments:
        return {
            "status": "insufficient_evidence",
            "version": LEARNING_ANALYSIS_VERSION,
            "method": "evidence_rules",
            "message": "판서 페이지와 음성 전사 자료가 없어 분석할 수 없습니다.",
            "pages": [], "concepts": [],
        }

    output_pages = []
    concept_index: dict[str, dict] = {}
    for page in pages:
        page_id = page.get("id")
        start = float(page.get("start", 0))
        end = float(page.get("end", start))
        page_segments = [
            segment for segment in segments
            if float(segment.get("end", 0)) > start and float(segment.get("start", 0)) < end
        ]
        ocr_page = ocr_pages.get(page_id)
        board_text, regions = _page_text(ocr_page)
        has_ocr = bool(board_text.strip())
        # Prefer readable OCR terms; transcript-only terms are a fallback
        # when OCR returned no confidently readable text.
        if has_ocr:
            board_terms = _extract_terms(board_text, include_pairs=True)
            terms = board_terms[:12]
        else:
            terms = _extract_terms("\n".join(str(s.get("text", "")) for s in page_segments), include_pairs=False)[:10]

        # Find all speech occurrences, including explanations outside the page's
        # exact display interval, when they explicitly mention a written concept.
        concept_occurrences = {}
        for term in terms:
            occurrences = []
            for segment in segments:
                if _matches(term, segment.get("text", "")):
                    occurrences.append({
                        "segment_id": segment.get("id"),
                        "start": float(segment.get("start", 0)),
                        "end": float(segment.get("end", 0)),
                        "text": str(segment.get("text", "")),
                    })
            concept_occurrences[_normal(term)] = occurrences

        links = []
        for segment in segments:
            overlap = float(segment.get("end", 0)) > start and float(segment.get("start", 0)) < end
            matched = [term for term in terms if _matches(term, segment.get("text", ""))]
            if not overlap and not matched:
                continue
            if overlap and matched:
                relation = "시간 구간 겹침 + 개념 일치"
            elif matched:
                relation = "필기 개념 언급"
            else:
                relation = "판서 시간 구간"
            links.append({
                "segment_id": segment.get("id"),
                "start": float(segment.get("start", 0)),
                "end": float(segment.get("end", 0)),
                "text": str(segment.get("text", "")),
                "relation": relation,
                "matched_concepts": matched,
            })

        linked_ids = {item["segment_id"] for item in links if item.get("segment_id")}
        scoring_segments = [
            segment for segment in segments if segment.get("id") in linked_ids
        ]
        score, signals, reasons = _score_page(scoring_segments, terms, regions, has_ocr)
        if score is None:
            level = "unrated"
        elif score >= 65:
            level = "high"
        elif score >= 35:
            level = "medium"
        else:
            level = "low"

        page_concepts = []
        for term in terms:
            normalized = _normal(term)
            if not normalized:
                continue
            occurrences = concept_occurrences.get(normalized, [])
            cue_count, cue_reasons = _cue_counts(occurrences)
            if segments:
                repeat_signal = min(1.0, len(occurrences) / 4.0)
                emphasis_signal = min(1.0, cue_count / 2.0)
                concept_score = round(100 * (0.62 * repeat_signal + 0.38 * emphasis_signal))
            else:
                concept_score = None
            region_matches = [
                region for region in regions
                if _matches(term, region.get("text", ""))
            ]
            board_entry = {
                "board_page_id": page_id,
                "start": start,
                "end": end,
                "text": board_text,
                "ocr_scores": [float(region.get("score", 0)) for region in region_matches],
            }
            page_concept = {
                "name": term,
                "importance_score": concept_score,
                "speech_occurrences": occurrences,
                "board_appearances": [board_entry] if has_ocr else [],
                "board_page_ids": [page_id] if has_ocr else [],
                "reasons": cue_reasons + ([f"음성에서 {len(occurrences)}회 언급"] if occurrences else []),
            }
            page_concepts.append(page_concept)
            existing = concept_index.get(normalized)
            if existing is None:
                existing = {
                    "id": "c" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10],
                    "name": term,
                    "importance_score": concept_score,
                    "speech_occurrences": [],
                    "board_appearances": [],
                    "board_page_ids": [],
                    "reasons": [],
                }
                concept_index[normalized] = existing
            if concept_score is not None:
                old_score = existing.get("importance_score")
                existing["importance_score"] = max(old_score or 0, concept_score)
            existing["speech_occurrences"].extend(occurrences)
            existing["board_appearances"].extend(page_concept["board_appearances"])
            if has_ocr and page_id not in existing["board_page_ids"]:
                existing["board_page_ids"].append(page_id)
            existing["reasons"].extend(page_concept["reasons"])

        output_pages.append({
            "board_page_id": page_id,
            "start": start,
            "end": end,
            "importance_score": score,
            "importance_level": level,
            "signals": signals,
            "reasons": reasons,
            "concepts": page_concepts,
            "transcript_links": links,
            "ocr_available": has_ocr,
            "speech_available": bool(segments),
        })

    concepts = list(concept_index.values())
    for concept in concepts:
        concept["speech_occurrences"] = sorted(
            {item["segment_id"]: item for item in concept["speech_occurrences"] if item.get("segment_id")}.values(),
            key=lambda item: item["start"],
        )
        concept["board_appearances"] = sorted(
            concept["board_appearances"], key=lambda item: item["start"]
        )
        concept["reasons"] = list(dict.fromkeys(concept["reasons"]))
    concepts.sort(
        key=lambda item: (item.get("importance_score") is not None,
                          item.get("importance_score") or -1,
                          len(item["speech_occurrences"])),
        reverse=True,
    )

    has_readable_ocr = any(_page_text(item)[0].strip() for item in ocr_pages.values())
    status = "complete" if segments and has_readable_ocr else "partial" if segments or has_readable_ocr else "insufficient_evidence"
    messages = []
    if not segments:
        messages.append("음성 전사 또는 자막이 없어 교수자 설명과 중요도 신호를 분석하지 못했습니다.")
    if not has_readable_ocr:
        messages.append("신뢰 가능한 OCR 텍스트가 없어 판서와 음성의 직접 개념 연결은 제한됩니다.")
    return {
        "status": status,
        "version": LEARNING_ANALYSIS_VERSION,
        "method": "evidence_rules",
        "message": " ".join(messages) if messages else "필기·음성의 시간 및 텍스트 근거를 연결했습니다.",
        "importance_scale": {"high": 65, "medium": 35},
        "pages": output_pages,
        "concepts": concepts,
    }


def build_learning_analysis(root, manifest: dict, force: bool = False) -> dict:
    """Cache the analysis separately from OCR, transcript, and page generation."""
    page_signature = [
        [page.get("id"), page.get("start"), page.get("end")]
        for page in (manifest.get("board_pages") or manifest.get("board_states") or [])
    ]
    transcript_signature = [
        [item.get("id"), item.get("start"), item.get("end"), item.get("text")]
        for item in (manifest.get("transcript") or {}).get("segments", [])
    ]
    ocr_signature = [
        [page.get("board_page_id"), [(region.get("text"), region.get("score")) for region in page.get("regions", [])]]
        for page in (manifest.get("ocr") or {}).get("pages", [])
    ]
    fingerprint = [
        LEARNING_ANALYSIS_VERSION,
        (manifest.get("source") or {}).get("sha256"),
        page_signature,
        transcript_signature,
        ocr_signature,
    ]
    return stage(
        root, "learning-analysis", fingerprint,
        lambda _folder: analyze_learning(manifest), force=force,
    )
