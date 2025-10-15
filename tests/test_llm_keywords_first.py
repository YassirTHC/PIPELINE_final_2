import importlib.machinery
import json
import logging
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest

from pipeline_core import llm_service
from pipeline_core import fetchers as pipeline_fetchers


if "cv2" not in sys.modules:
    cv2_stub = ModuleType("cv2")
    cv2_stub.cvtColor = lambda *args, **kwargs: None
    cv2_stub.Canny = lambda *args, **kwargs: []
    cv2_stub.findContours = lambda *args, **kwargs: ([], [])
    cv2_stub.moments = lambda *args, **kwargs: []
    cv2_stub.contourArea = lambda *args, **kwargs: 0
    cv2_stub.resize = lambda *args, **kwargs: None
    cv2_stub.COLOR_RGB2GRAY = 0
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 0
    cv2_stub.INTER_LANCZOS4 = 0
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

if "moviepy.editor" not in sys.modules:
    moviepy_module = ModuleType("moviepy")
    editor_stub = ModuleType("moviepy.editor")

    class _ClipStub:  # pragma: no cover - lightweight placeholder
        def __init__(self, *args, **kwargs):
            pass

    editor_stub.VideoFileClip = _ClipStub
    editor_stub.TextClip = _ClipStub
    editor_stub.CompositeVideoClip = _ClipStub
    moviepy_module.VideoFileClip = _ClipStub
    moviepy_module.TextClip = _ClipStub
    moviepy_module.CompositeVideoClip = _ClipStub
    moviepy_module.editor = editor_stub
    sys.modules["moviepy"] = moviepy_module
    sys.modules["moviepy.editor"] = editor_stub

import video_processor


def test_keywords_prompt_schema():
    prompt = llm_service._build_keywords_prompt("Example transcript", "en")
    assert "Return ONLY one JSON object with keys: broll_keywords, queries." in prompt
    assert "broll_keywords: 8-12 visual noun phrases" in prompt
    assert "queries: 8-12 short, filmable search queries" in prompt
    assert "hashtags" not in prompt.lower()


def test_json_metadata_prompt_mentions_viral_requirements():
    prompt = llm_service._build_json_metadata_prompt("Segment sur la motivation et la discipline")
    assert "Answer strictly in English." in prompt
    assert "Return ONLY one JSON object with the exact keys" in prompt
    assert "ultra-hook headline" in prompt


def test_non_empty_keywords(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "1")

    payload = {
        "title": "Short Title",
        "description": "Short description",
        "queries": ["studio lighting"],
        "broll_keywords": ["modern office"],
    }

    def fake_generate(prompt: str, **kwargs):
        return payload, {"response": json.dumps(payload)}, len(json.dumps(payload))

    monkeypatch.setattr(llm_service, "_ollama_generate_json", fake_generate)

    transcript = "Marketing teams discuss growth strategies, customer journeys, analytics dashboards, and product demos."
    result = llm_service.generate_metadata_as_json(transcript)

    assert result["title"] == "Short Title"
    assert result["description"] == "Short description"
    assert len(result["queries"]) >= 8
    assert len(result["broll_keywords"]) >= 8
    assert result["queries"][0] == "studio lighting"
    assert result["broll_keywords"][0] == "modern office"


def test_secondary_prompt_recovers_summary(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")
    monkeypatch.delenv("PIPELINE_LLM_DISABLE_HASHTAGS", raising=False)
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "0")

    keywords_payload = {
        "broll_keywords": [
            "reward journal writing",
            "personal milestone tracker",
            "focused breathing practice",
            "motivational vision board",
            "celebration fist pump",
            "goal progress checklist",
            "calm morning planning",
            "intrinsic motivation walk",
        ],
        "queries": [
            "personal motivation focus",
            "self reward routine",
            "intrinsic drive habits",
            "goal celebration moment",
            "progress tracking journal",
            "mindful reward ritual",
            "achievement reflection time",
            "positive reinforcement habit",
        ],
    }

    def fake_generate(prompt: str, **kwargs):
        raw = json.loads(json.dumps(keywords_payload))
        return raw, raw, len(json.dumps(raw))

    fallback_payload = {
        "title": "Intrinsic Motivation Momentum",
        "description": "Build unstoppable discipline by celebrating every personal win.",
        "hashtags": [
            "#motivation",
            "#selfgrowth",
            "#productivity",
            "#mindset",
            "#success",
            "#habits",
        ],
        "broll_keywords": [
            "fallback reward moment",
            "fallback goal focus",
            "fallback journal writing",
            "fallback mindset reset",
            "fallback celebration smile",
            "fallback planning shot",
            "fallback reflection desk",
            "fallback sunrise walk",
        ],
        "queries": [
            "fallback reward focus",
            "fallback self discipline",
            "fallback motivation habit",
            "fallback progress ritual",
            "fallback success mindset",
            "fallback positive journey",
            "fallback goal tracker",
            "fallback achievement focus",
        ],
    }

    def fake_json(prompt: str, **kwargs):
        return json.loads(json.dumps(fallback_payload))

    monkeypatch.setattr(llm_service, "_ollama_generate_json", fake_generate)
    monkeypatch.setattr(llm_service, "_ollama_json", fake_json)

    transcript = "Celebrating small wins keeps internal motivation strong and builds powerful habits over time."
    result = llm_service.generate_metadata_as_json(transcript)

    assert result["title"] == fallback_payload["title"]
    assert result["description"] == fallback_payload["description"]
    assert result["hashtags"][: len(fallback_payload["hashtags"])] == fallback_payload["hashtags"]
    assert result["queries"] == keywords_payload["queries"]
    assert result["broll_keywords"] == keywords_payload["broll_keywords"]


def test_keywords_first_failure_falls_back_to_rich_prompt(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")

    def failing_generate(*_args, **_kwargs):
        raise RuntimeError("timeout")

    fallback_payload = {
        "title": "Viral Discipline Hack",
        "description": "Reward your grind with mini wins ✨ Keep momentum high!",
        "hashtags": [
            "#motivation",
            "#discipline",
            "#grindmode",
            "#selfgrowth",
            "#successmindset",
            "#goalcrusher",
        ],
        "broll_keywords": [
            "motivated runner",
            "victory fist pump",
            "habit tracker journal",
            "focus breathing shot",
            "goal board closeup",
            "sunrise training",
            "celebration smile",
            "progress checklist",
        ],
        "queries": [
            "runner celebrating finish",
            "writing habit journal",
            "morning focus routine",
            "success celebration moment",
            "discipline motivation shot",
            "goal tracker closeup",
            "motivational pep talk",
            "victory celebration steps",
        ],
    }

    monkeypatch.setattr(llm_service, "_ollama_generate_json", failing_generate)
    monkeypatch.setattr(llm_service, "_ollama_json", lambda *_args, **_kwargs: fallback_payload)

    transcript = "Keep rewarding your effort to stay motivated and unstoppable."
    result = llm_service.generate_metadata_as_json(transcript)

    assert result["title"] == fallback_payload["title"]
    assert result["description"] == fallback_payload["description"]
    assert result["hashtags"][:6] == fallback_payload["hashtags"]
    assert result["queries"][:8] == fallback_payload["queries"][:8]
    assert result["broll_keywords"][:8] == fallback_payload["broll_keywords"][:8]


def test_generate_hints_for_segment_integrates(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")

    expected_queries = [
        "marketing analytics dashboard",
        "team planning session",
        "customer success meeting",
        "digital strategy workshop",
        "startup founders collaboration",
        "audience engagement presentation",
        "business growth chart",
        "product demo recording",
    ]
    expected_keywords = [
        "marketing team collaboration",
        "analytics dashboard closeup",
        "customer success handshake",
        "digital workshop whiteboard",
        "startup founders meeting",
        "audience engagement speaker",
        "business growth presentation",
        "product demo studio",
    ]

    captured = {}

    def fake_segment_json(self, snippet: str, timeout_s=None, num_predict=None):
        captured["snippet"] = snippet
        return {
            "title": "Segment Title",
            "description": "Segment Description",
            "queries": expected_queries,
            "broll_keywords": expected_keywords,
        }

    monkeypatch.setattr(
        llm_service.LLMMetadataGeneratorService,
        "_segment_llm_json",
        fake_segment_json,
        raising=False,
    )

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    result = service.generate_hints_for_segment(
        "The host explains marketing analytics and customer growth strategies with vivid examples.",
        0.0,
        10.0,
    )

    assert "marketing analytics" in captured["snippet"]
    assert result["title"] == "Segment Title"
    assert result["description"] == "Segment Description"
    assert result["queries"] == expected_queries
    assert result["broll_keywords"] == expected_keywords
    assert isinstance(result.get("filters"), dict)


def test_service_exposes_call_llm_and_fallbacks(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_KEYWORDS",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_QUERIES",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )

    def fail_call_llm(self, prompt: str, max_tokens: int = 192):
        raise ValueError("timeout")

    monkeypatch.setattr(
        llm_service.LLMMetadataGeneratorService,
        "_call_llm",
        fail_call_llm,
        raising=False,
    )

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    result = service.generate_hints_for_segment(
        "Marketing teams review analytics dashboards, share growth metrics, and plan new campaigns.",
        0.0,
        10.0,
    )

    queries = [query for query in result.get("queries", []) if query]
    assert len(queries) >= 4


def test_generate_hints_handles_invalid_json_with_metadata_fallback(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")

    def broken_call_llm(self, prompt: str, max_tokens: int = 192):
        return "{not-valid-json"

    monkeypatch.setattr(
        llm_service.LLMMetadataGeneratorService,
        "_call_llm",
        broken_call_llm,
        raising=False,
    )

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    service.last_metadata = {
        "queries": [
            "city skyline timelapse",
            "downtown office teamwork",
        ],
        "broll_keywords": [
            "city skyline",
            "office teamwork",
        ],
    }

    result = service.generate_hints_for_segment(
        "Marketing teams discuss analytics and customer journeys.",
        12.0,
        20.0,
    )

    queries = [query for query in result.get("queries", []) if query]
    assert queries, "expected fallback queries from metadata"
    assert result.get("source") == "metadata_keywords_fallback"
    assert all(" " in query for query in queries)


def test_generate_hints_handles_empty_llm_response_with_metadata(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")

    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_QUERIES",
        {"values": ["sunset beach"], "updated_at": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_KEYWORDS",
        {"values": ["calm ocean"], "updated_at": 0.0},
        raising=False,
    )

    def empty_call_llm(self, prompt: str, max_tokens: int = 192):
        return ""

    monkeypatch.setattr(
        llm_service.LLMMetadataGeneratorService,
        "_call_llm",
        empty_call_llm,
        raising=False,
    )

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    service.last_metadata = {
        "queries": ["sunset beach"],
        "broll_keywords": ["calm ocean"],
    }

    result = service.generate_hints_for_segment(
        "The narrator describes relaxing beach visuals and ocean waves.",
        0.0,
        5.0,
    )

    queries = [query for query in result.get("queries", []) if query]
    assert queries, "expected metadata fallback queries"
    assert result.get("source") == "metadata_keywords_fallback"


def test_disable_hashtags_env_clears_result(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "1")
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_KEYWORDS",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_QUERIES",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )

    payload = {
        "title": "Campaign Strategy",
        "description": "A quick summary of the marketing campaign roadmap.",
        "hashtags": ["#GrowthMarketing", "#CampaignLaunch"],
        "broll_keywords": [
            "marketing team meeting",
            "analytics dashboard review",
            "digital strategy planning",
            "startup founders collaboration",
            "customer journey mapping",
            "social media scheduling",
        ],
        "queries": [
            "marketing team collaboration",
            "analytics dashboard presentation",
            "digital strategy meeting",
            "startup founders planning",
            "customer journey workshop",
            "social media strategy session",
        ],
    }

    def fake_generate_json(*args, **kwargs):
        serialized = json.dumps(payload)
        return payload, {"response": serialized}, len(serialized)

    monkeypatch.setattr(llm_service, "_ollama_generate_json", fake_generate_json)

    result = llm_service.generate_metadata_as_json(
        "The host outlines a marketing campaign, reviews analytics, and discusses customer journeys."
    )

    assert result["hashtags"] == []


def test_segment_processing_uses_hint_queries_and_logs_candidate_event(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "1")

    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    clips_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    monkeypatch.setattr(video_processor.Config, "CLIPS_FOLDER", clips_dir, raising=False)
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", output_dir, raising=False)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", temp_dir, raising=False)
    monkeypatch.setattr(
        video_processor.whisper,
        "load_model",
        lambda *_args, **_kwargs: SimpleNamespace(transcribe=lambda *a, **k: {}),
    )

    processor = video_processor.VideoProcessor()

    events = []

    class _DummyLogger:
        def log(self, payload):
            events.append(dict(payload))

    event_logger = _DummyLogger()
    monkeypatch.setattr(processor, "_get_broll_event_logger", lambda: event_logger)

    class _HintService:
        def generate_hints_for_segment(self, text, start, end):
            queries = [
                "marketing analytics dashboard",
                "team planning session",
                "customer journey mapping",
            ]
            source = "llm_segment"
            logging.getLogger("pipeline_core.llm_service").info(
                "[BROLL][LLM] segment=%s queries=%s (source=%s)",
                f"{start:.2f}-{end:.2f}",
                queries[:4],
                source,
            )
            return {
                "queries": queries,
                "broll_keywords": [
                    "analytics dashboard",
                    "team planning",
                ],
                "source": source,
                "filters": {"orientation": "landscape"},
            }

        def provider_fallback_queries(self, *_args, **_kwargs):
            return [], "none"

    processor._llm_service = _HintService()
    processor._derive_segment_keywords = MethodType(
        lambda self, _segment, _keywords: ["analytics", "planning"],
        processor,
    )
    processor._rank_candidate = MethodType(lambda self, *_args, **_kwargs: 0.0, processor)
    processor._download_core_candidate = MethodType(
        lambda self, *_args, **_kwargs: None,
        processor,
    )

    recorded = {}

    def fake_fetch(
        self,
        queries,
        *,
        segment_index=None,
        duration_hint=None,
        filters=None,
        segment_timeout_s=None,
    ):
        recorded["queries"] = list(queries)
        recorded["segment_index"] = segment_index
        event_logger.log(
            {
                "event": "broll_candidate_evaluated",
                "provider": "stub",
                "segment": segment_index,
                "count": len(recorded["queries"]),
            }
        )
        return []

    monkeypatch.setattr(
        pipeline_fetchers.FetcherOrchestrator,
        "fetch_candidates",
        fake_fetch,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_fetchers.FetcherOrchestrator,
        "evaluate_candidate_filters",
        lambda self, *_args, **_kwargs: (True, None),
        raising=False,
    )

    segment = SimpleNamespace(start=0.0, end=4.0, text="Discussing marketing analytics and customer journeys")

    with caplog.at_level("INFO", logger="pipeline_core.llm_service"):
        processor._insert_brolls_pipeline_core(
            [segment],
            ["analytics", "planning"],
            subtitles=None,
            input_path=tmp_path / "input.mp4",
        )

    assert recorded.get("queries"), "expected orchestrator to receive queries"
    assert all(" " in query for query in recorded["queries"])

    candidate_events = [event for event in events if event.get("event") == "broll_candidate_evaluated"]
    assert candidate_events, "expected candidate evaluation telemetry"

    log_lines = [record.getMessage() for record in caplog.records]
    assert any("[BROLL][LLM] segment=0.00-4.00" in line for line in log_lines), "expected LLM hint log"
    assert any("marketing analytics dashboard" in line for line in log_lines)
    assert any("source=llm_segment" in line for line in log_lines)

