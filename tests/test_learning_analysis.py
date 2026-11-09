from lecture2doc.learning_analysis import analyze_learning


def test_links_board_concepts_to_speech_and_keeps_each_appearance_time():
    manifest = {
        "board_pages": [{"id": "p00001", "start": 10.0, "end": 20.0}],
        "transcript": {
            "status": "complete",
            "segments": [
                {"id": "t000001", "start": 12.0, "end": 15.0, "text": "선형 변환의 정의를 알아두세요."},
                {"id": "t000002", "start": 40.0, "end": 43.0, "text": "선형변환은 시험에 자주 나옵니다."},
                {"id": "t000003", "start": 50.0, "end": 54.0, "text": "다음은 행렬 곱셈입니다."},
            ],
        },
        "ocr": {
            "pages": [{
                "board_page_id": "p00001",
                "regions": [{"text": "선형 변환", "score": 0.98}],
            }],
        },
    }

    result = analyze_learning(manifest)
    page = result["pages"][0]
    concept = next(item for item in result["concepts"] if item["name"] == "선형 변환")

    assert result["status"] == "complete"
    assert page["importance_score"] is not None
    assert page["importance_level"] == "high"
    assert {link["segment_id"] for link in page["transcript_links"]} >= {"t000001", "t000002"}
    assert {item["segment_id"] for item in concept["speech_occurrences"]} == {"t000001", "t000002"}
    assert concept["board_appearances"][0]["start"] == 10.0


def test_transcript_only_analysis_is_partial_and_uses_page_time():
    manifest = {
        "board_pages": [{"id": "p00002", "start": 5.0, "end": 18.0}],
        "transcript": {
            "segments": [
                {"id": "t000010", "start": 6.0, "end": 9.0, "text": "고유값의 정의를 설명하겠습니다."},
                {"id": "t000011", "start": 12.0, "end": 15.0, "text": "고유값은 중요하니 기억하세요."},
            ],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["importance_score"] is not None
    assert any(item["name"] == "고유값" for item in result["concepts"])


def test_ocr_without_speech_does_not_invent_importance_score():
    manifest = {
        "board_pages": [{"id": "p00003", "start": 2.0, "end": 12.0}],
        "transcript": {"segments": []},
        "ocr": {
            "pages": [{
                "board_page_id": "p00003",
                "regions": [{"text": "미분 방정식", "score": 0.91}],
            }],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["importance_score"] is None
    assert result["pages"][0]["importance_level"] == "unrated"
    assert result["concepts"][0]["board_appearances"][0]["start"] == 2.0


def test_no_board_or_transcript_returns_insufficient_evidence():
    result = analyze_learning({})

    assert result["status"] == "insufficient_evidence"
    assert result["pages"] == []
    assert result["concepts"] == []


def test_low_confidence_ocr_text_is_not_treated_as_board_evidence():
    manifest = {
        "board_pages": [{"id": "p00004", "start": 2.0, "end": 12.0}],
        "transcript": {
            "segments": [
                {"id": "t000020", "start": 3.0, "end": 6.0, "text": "고유값은 중요하니 기억하세요."},
            ],
        },
        "ocr": {
            "pages": [{
                "board_page_id": "p00004",
                "regions": [{"text": "고유값", "score": 0.45, "review": True}],
            }],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["ocr_available"] is False
    assert result["pages"][0]["importance_score"] is not None
    concept = next(item for item in result["concepts"] if item["name"] == "고유값")
    assert concept["board_appearances"] == []
