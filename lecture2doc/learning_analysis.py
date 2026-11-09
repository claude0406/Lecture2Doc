"""Evidence-based lecture importance and AI-curated concept appearance times."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request

from .storage import stage

LEARNING_ANALYSIS_VERSION = "2026-11-10.learning-importance-2-ai"
_TOKEN = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?|[∫∑∏√∂∆∇∞≤≥≠≈→←↔⊂∈]+")
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
    "것", "것도", "수", "번", "한번", "한번더", "일단", "그냥", "약간", "조금",
    "정말", "매우", "굉장히", "그러면", "그러니까", "그럼", "그런데",
    "저희", "우리", "여기서", "이렇게요", "맞아요", "아니", "예", "네",
    "이런", "저런", "무슨", "그게", "이게", "저게", "the", "this", "that",
    "with", "from", "and", "then", "what", "when", "where", "which", "about",
}
_IMPORTANCE_CUES = (
    ("시험·출제 언급", re.compile(r"시험|출제|기출")),
    ("중요하다는 강조", re.compile(r"중요|핵심|핵심적")),
    ("암기·기억 권고", re.compile(r"기억|외우|암기|잊지")),
    ("주의·오류 경고", re.compile(r"주의|조심|틀리|실수|헷갈")),
    ("정의·원리 설명", re.compile(r"정의|원리|정리하면|공식")),
    ("유도·증명 과정", re.compile(r"유도|증명|도출")),
    ("대표 문제·예제", re.compile(r"예제|대표 문제|연습문제|문제 풀")),
)
_MAX_CONCEPTS = 8
_AI_MODEL_DEFAULT = "gpt-6-luna"


def _normal(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣∫∑∏√∂∆∇∞≤≥≠≈→←↔⊂∈]+", "", str(value).lower())


def _tokens(value: str) -> list[str]:
    output = []
    for token in _TOKEN.findall(str(value)):
        token = token.strip().lower()
        if token in _STOPWORDS:
            continue
        if re.fullmatch(r"[가-힣]+", token):
            for particle in _KOREAN_PARTICLES:
                if token.endswith(particle) and len(token) - len(particle) >= 2:
                    token = token[:-len(particle)]
                    break
        if token in _STOPWORDS:
            continue
        if token.isascii() and token.isalpha() and len(token) < 2:
            continue
        if token.isdigit() or len(token) < 2:
            continue
        output.append(token)
    return output


def _page_text(ocr_page: dict | None) -> tuple[str, list[dict]]:
    if not ocr_page:
        return "", []
    all_regions = ocr_page.get("regions") or []
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
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", raw_text))
    return needle in _normal(raw_text)


def _extract_terms(text: str, include_pairs: bool = False) -> list[str]:
    words = _tokens(text)
    found = list(words)
    if include_pairs:
        found.extend(
            f"{left} {right}" for left, right in zip(words, words[1:])
            if len(_normal(f"{left}{right}")) >= 4
        )
    unique = {}
    for term in found:
        key = _normal(term)
        if len(key) >= 2:
            unique.setdefault(key, term)
    return list(unique.values())


def _score_page(segments: list[dict], terms: list[str], has_ocr: bool) -> tuple[int | None, dict, list[str]]:
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
    weights = [(0.40, emphasis), (0.30, repetition)]
    if has_ocr:
        weights.append((0.22, alignment))
        weights.append((0.08, 1.0 if terms else 0.0))
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
        "emphasis": round(emphasis, 3),
        "repetition": round(repetition, 3),
        "board_speech_alignment": round(alignment, 3) if has_ocr else None,
        "board_concepts": len(terms),
        "emphasis_cue_count": cue_count,
        "matched_speech_segments": len(matched_segments),
        "repeated_term_mentions": repeated_term_mentions,
    }, reasons


def analyze_learning(manifest: dict) -> dict:
    """Build page importance and exact board/speech concept occurrences."""
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
        board_text, regions = _page_text(ocr_pages.get(page_id))
        has_ocr = bool(board_text.strip())
        terms = (
            _extract_terms(board_text, include_pairs=True)[:12]
            if has_ocr else
            _extract_terms("\n".join(str(s.get("text", "")) for s in page_segments))[:10]
        )

        concept_occurrences = {}
        for term in terms:
            concept_occurrences[_normal(term)] = [
                {
                    "segment_id": segment.get("id"),
                    "start": float(segment.get("start", 0)),
                    "end": float(segment.get("end", 0)),
                    "text": str(segment.get("text", "")),
                }
                for segment in segments
                if _matches(term, segment.get("text", ""))
            ]

        scoring_segments = [
            segment for segment in segments
            if (
                float(segment.get("end", 0)) > start
                and float(segment.get("start", 0)) < end
            ) or any(_matches(term, segment.get("text", "")) for term in terms)
        ]
        score, signals, reasons = _score_page(scoring_segments, terms, has_ocr)
        level = "unrated" if score is None else "high" if score >= 65 else "medium" if score >= 35 else "low"

        page_concepts = []
        for term in terms:
            normalized = _normal(term)
            if not normalized:
                continue
            occurrences = concept_occurrences.get(normalized, [])
            cue_count, cue_reasons = _cue_counts(occurrences)
            concept_score = (
                round(100 * (0.62 * min(1.0, len(occurrences) / 4.0) + 0.38 * min(1.0, cue_count / 2.0)))
                if segments else None
            )
            region_matches = [region for region in regions if _matches(term, region.get("text", ""))]
            board_entry = {
                "board_page_id": page_id, "start": start, "end": end,
                "text": board_text,
                "ocr_scores": [float(region.get("score", 0)) for region in region_matches],
            }
            page_concept = {
                "name": term, "importance_score": concept_score,
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
                    "name": term, "importance_score": concept_score,
                    "speech_occurrences": [], "board_appearances": [],
                    "board_page_ids": [], "reasons": [],
                }
                concept_index[normalized] = existing
            if concept_score is not None:
                existing["importance_score"] = max(existing.get("importance_score") or 0, concept_score)
            existing["speech_occurrences"].extend(occurrences)
            existing["board_appearances"].extend(page_concept["board_appearances"])
            if has_ocr and page_id not in existing["board_page_ids"]:
                existing["board_page_ids"].append(page_id)
            existing["reasons"].extend(page_concept["reasons"])

        output_pages.append({
            "board_page_id": page_id, "start": start, "end": end,
            "importance_score": score, "importance_level": level,
            "signals": signals, "reasons": reasons, "concepts": page_concepts,
            "ocr_available": has_ocr, "speech_available": bool(segments),
        })

    concepts = list(concept_index.values())
    for concept in concepts:
        concept["speech_occurrences"] = sorted(
            {item["segment_id"]: item for item in concept["speech_occurrences"] if item.get("segment_id")}.values(),
            key=lambda item: item["start"],
        )
        concept["board_appearances"] = sorted(concept["board_appearances"], key=lambda item: item["start"])
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
        "status": status, "version": LEARNING_ANALYSIS_VERSION,
        "method": "evidence_rules",
        "message": " ".join(messages) if messages else "필기와 음성의 시간·텍스트 근거를 분석했습니다.",
        "importance_scale": {"high": 65, "medium": 35},
        "pages": output_pages, "concepts": concepts,
    }


def _sample_transcript(segments: list[dict], limit: int = 16000) -> str:
    lines = [
        f"[{float(item.get('start', 0)):.1f}-{float(item.get('end', 0)):.1f}] "
        + re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        for item in segments
    ]
    if sum(len(line) + 1 for line in lines) <= limit:
        return "\n".join(lines)
    target_count = min(len(lines), max(1, limit // 180))
    indexes = sorted({
        round(i * (len(lines) - 1) / max(1, target_count - 1))
        for i in range(target_count)
    })
    selected, used = [], 0
    for index in indexes:
        line = lines[index][:240]
        if used + len(line) + 1 > limit:
            break
        selected.append(line)
        used += len(line) + 1
    return "\n".join(selected)


def _ai_request(payload: dict, api_key: str, model: str) -> dict:
    """Send transcript and board text to OpenAI for structured concept selection."""
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "source_terms": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                        "relevance": {"type": "integer"},
                    },
                    "required": ["name", "source_terms", "reason", "relevance"],
                },
            },
        },
        "required": ["topic", "concepts"],
    }
    body = {
        "model": model, "store": False,
        "instructions": (
            "당신은 대학 강의의 핵심 개념을 선별하는 교육 분석가입니다. "
            "전사문과 판서 OCR로 실제 강의 주제를 파악하고 학습 가치가 높은 구체 개념을 최대 8개 고르세요. "
            "말버릇, 조사, 대명사, 일반 표현(한번, 것도, 이거, 이제, 내용, 부분, 설명)은 개념으로 선택하지 마세요. "
            "수학·과학·공학 용어, 정의, 법칙, 정리, 알고리즘, 공식, 핵심 방법을 우선하세요. "
            "source_terms에는 후보 목록에 실제 있는 표현만 쓰고 원문에 없는 개념을 만들지 마세요. "
            "동일 개념의 띄어쓰기나 표현 변형은 하나로 묶으세요. 근거가 약하면 적게 선택하거나 빈 배열을 반환하세요. "
            "reason은 강의에서 중요한 이유를 한국어 한 문장으로 작성하세요. 반드시 JSON 형식으로 답하세요."
        ),
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {"format": {
            "type": "json_schema", "name": "lecture_concept_selection",
            "strict": True, "schema": schema,
        }},
        "max_output_tokens": 1800,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=75) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API 오류({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"OpenAI API 연결 실패: {exc}") from exc

    text_parts = []
    for output in response_data.get("output", []):
        if output.get("type") == "message":
            text_parts.extend(
                content["text"] for content in output.get("content", [])
                if content.get("type") == "output_text" and content.get("text")
            )
    if not text_parts:
        raise RuntimeError("AI 응답에서 개념 선별 결과를 읽지 못했습니다.")
    try:
        parsed = json.loads("\n".join(text_parts))
    except json.JSONDecodeError as exc:
        raise RuntimeError("AI 응답이 올바른 JSON 형식이 아닙니다.") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("concepts"), list):
        raise RuntimeError("AI 응답에 concepts 목록이 없습니다.")
    return parsed


def _fallback_concepts(concepts: list[dict]) -> list[dict]:
    selected, seen = [], set()
    for item in concepts:
        name = str(item.get("name", "")).strip()
        normalized = _normal(name)
        if not normalized or normalized in seen or not _tokens(name):
            continue
        if not item.get("board_page_ids") and len(item.get("speech_occurrences", [])) < 2:
            continue
        selected.append(dict(item, ai_reason="기본 필터: 판서 또는 반복 발언에서 확인된 용어"))
        seen.add(normalized)
        if len(selected) >= _MAX_CONCEPTS:
            break
    return selected


def _apply_ai_selection(result: dict, selection: dict, model: str) -> dict:
    candidates = result.get("concepts", [])
    by_name = {_normal(item.get("name", "")): item for item in candidates}
    chosen, used = [], set()
    for entry in selection.get("concepts", [])[:_MAX_CONCEPTS]:
        if not isinstance(entry, dict):
            continue
        aliases = entry.get("source_terms") or [entry.get("name", "")]
        matched = []
        for alias in aliases:
            item = by_name.get(_normal(alias))
            if item and item["id"] not in used:
                matched.append(item)
        if not matched:
            continue
        used.update(item["id"] for item in matched)
        merged = dict(matched[0])
        if len(matched) > 1:
            merged["speech_occurrences"] = sorted(
                {row.get("segment_id"): row for item in matched for row in item.get("speech_occurrences", []) if row.get("segment_id")}.values(),
                key=lambda row: row["start"],
            )
            merged["board_appearances"] = sorted(
                {row.get("board_page_id"): row for item in matched for row in item.get("board_appearances", [])}.values(),
                key=lambda row: row["start"],
            )
            merged["board_page_ids"] = list(dict.fromkeys(
                page_id for item in matched for page_id in item.get("board_page_ids", [])
            ))
            scores = [item.get("importance_score") for item in matched if item.get("importance_score") is not None]
            merged["importance_score"] = max(scores) if scores else None
            merged["reasons"] = list(dict.fromkeys(
                reason for item in matched for reason in item.get("reasons", [])
            ))
        merged["name"] = str(entry.get("name") or merged["name"]).strip()[:48] or merged["name"]
        merged["source_terms"] = [item["name"] for item in matched]
        merged["ai_reason"] = str(entry.get("reason", "")).strip()[:180]
        try:
            merged["relevance_score"] = max(0, min(100, int(entry.get("relevance", 0))))
        except (TypeError, ValueError):
            merged["relevance_score"] = 0
        chosen.append(merged)

    result = dict(result)
    result["concepts"] = chosen
    selected_terms = {
        _normal(term) for item in chosen
        for term in item.get("source_terms", [item.get("name", "")])
    }
    for page in result.get("pages", []):
        page["concepts"] = [
            item for item in page.get("concepts", [])
            if _normal(item.get("name", "")) in selected_terms
        ]
    topic = str(selection.get("topic", "")).strip()[:100]
    result["ai_curation"] = {
        "status": "complete", "provider": "openai", "model": model,
        "topic": topic, "selected_count": len(chosen),
        "message": "AI가 강의 문맥과 판서 근거를 바탕으로 핵심 개념을 선별했습니다.",
    }
    result["method"] = "evidence_rules+ai_curated"
    result["message"] = (
        f"강의 주제: {topic}. AI가 핵심 개념 {len(chosen)}개를 선별했습니다."
        if topic else f"AI가 핵심 개념 {len(chosen)}개를 선별했습니다."
    )
    return result


def _curate_concepts_with_ai(manifest: dict, result: dict) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("LECTURE2DOC_AI_MODEL", _AI_MODEL_DEFAULT).strip() or _AI_MODEL_DEFAULT

    def fallback(status: str, message: str) -> dict:
        concepts = _fallback_concepts(result.get("concepts", []))
        output = dict(result, concepts=concepts)
        selected = {_normal(item.get("name", "")) for item in concepts}
        for page in output.get("pages", []):
            page["concepts"] = [item for item in page.get("concepts", []) if _normal(item.get("name", "")) in selected]
        output["ai_curation"] = {
            "status": status, "provider": "openai" if status == "failed" else None,
            "model": model if status == "failed" else None,
            "topic": "", "selected_count": len(concepts), "message": message,
        }
        output["message"] = message
        output["method"] = "evidence_rules"
        return output

    if not api_key:
        return fallback(
            "not_configured",
            "AI 키가 없어 기본 필터로 개념을 추렸습니다. OPENAI_API_KEY를 설정하면 AI 문맥 선별을 사용할 수 있습니다.",
        )

    segments = (manifest.get("transcript") or {}).get("segments", [])
    candidates = [
        {
            "name": item["name"],
            "mentions": len(item.get("speech_occurrences", [])),
            "board_pages": len(item.get("board_page_ids", [])),
            "importance_score": item.get("importance_score"),
            "examples": [row.get("text", "")[:180] for row in item.get("speech_occurrences", [])[:2]],
        }
        for item in result.get("concepts", [])[:80]
    ]
    board_samples = []
    for page in (manifest.get("ocr") or {}).get("pages", []):
        text_value, _ = _page_text(page)
        if text_value.strip():
            board_samples.append(text_value.strip()[:800])
        if sum(len(value) for value in board_samples) >= 5000:
            break
    payload = {
        "task": "강의 문맥에 맞는 학습 핵심 개념 선별",
        "candidate_terms": candidates,
        "board_text_samples": "\n---\n".join(board_samples)[:5000],
        "transcript_sample": _sample_transcript(segments),
        "selection_limit": _MAX_CONCEPTS,
    }
    try:
        return _apply_ai_selection(result, _ai_request(payload, api_key, model), model)
    except Exception as exc:
        return fallback(
            "failed",
            f"AI 선별에 실패해 기본 필터 결과를 표시합니다. 오류: {type(exc).__name__}",
        )


def build_learning_analysis(root, manifest: dict, force: bool = False) -> dict:
    """Cache learning analysis separately from OCR and transcription."""
    page_signature = [
        [page.get("id"), page.get("start"), page.get("end")]
        for page in (manifest.get("board_pages") or manifest.get("board_states") or [])
    ]
    transcript_signature = [
        [item.get("id"), item.get("start"), item.get("end"), item.get("text")]
        for item in (manifest.get("transcript") or {}).get("segments", [])
    ]
    ocr_signature = [
        [page.get("board_page_id"), [(region.get("text"), region.get("score"), region.get("review")) for region in page.get("regions", [])]]
        for page in (manifest.get("ocr") or {}).get("pages", [])
    ]
    api_key_configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    model = os.environ.get("LECTURE2DOC_AI_MODEL", _AI_MODEL_DEFAULT).strip() or _AI_MODEL_DEFAULT
    fingerprint = [
        LEARNING_ANALYSIS_VERSION,
        (manifest.get("source") or {}).get("sha256"),
        page_signature, transcript_signature, ocr_signature,
        api_key_configured, model if api_key_configured else None,
    ]
    return stage(
        root, "learning-analysis", fingerprint,
        lambda _folder: _curate_concepts_with_ai(manifest, analyze_learning(manifest)),
        force=force,
    )
