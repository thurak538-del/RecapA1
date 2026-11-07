from types import SimpleNamespace
from unittest.mock import patch
import time

import pytest

from app.pipeline.gemini_rewriter import GeminiRewriter
from app.pipeline.recap_formula import build_story_context, run_recap_formula


class _RateLimitedModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED retry in 120s")


def _rewriter_with_models(models):
    rw = GeminiRewriter.__new__(GeminiRewriter)
    rw.clients = [SimpleNamespace(models=models)]
    rw._cur = 0
    rw._cooldown = {}
    rw._exhausted = set()
    rw._bad_keys = set()
    rw._process_deadline = None
    rw.progress_callback = None
    return rw


def test_gemini_rate_limit_moves_on_and_stops_at_generation_deadline(monkeypatch):
    models = _RateLimitedModels()
    rw = _rewriter_with_models(models)
    now = [0.0]
    monkeypatch.setattr(GeminiRewriter, "MAX_GENERATION_SECONDS", 10.0)

    with patch("app.pipeline.gemini_rewriter.time.monotonic", side_effect=lambda: now[0]), \
         patch("app.pipeline.gemini_rewriter.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
        with pytest.raises(RuntimeError, match="stopped after"):
            rw._generate("prompt", ["model-a", "model-b"])

    assert models.calls == 2
    assert now[0] == 10.0


def test_gemini_504_immediately_retries_same_model_request_on_next_key():
    class DeadlineModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("504 DEADLINE_EXCEEDED: operation took too long")

    class WorkingModels:
        def __init__(self):
            self.requests = []

        def generate_content(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(text='{"ok": true}')

    models = DeadlineModels()
    working = WorkingModels()
    rw = _rewriter_with_models(models)
    rw.clients.append(SimpleNamespace(models=working))

    with patch("app.pipeline.gemini_rewriter.time.sleep") as sleep:
        result = rw._generate("same large prompt", ["model-a", "model-b"])

    assert result == '{"ok": true}'
    assert models.calls == 1
    assert len(working.requests) == 1
    assert working.requests[0]["model"] == "model-a"
    assert working.requests[0]["contents"] == "same large prompt"
    assert (0, "model-a") in rw._cooldown
    sleep.assert_not_called()


def test_gemini_keeps_rotating_after_every_key_is_temporarily_busy():
    class RecoversOnSecondAttempt:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("504 DEADLINE_EXCEEDED: server busy")
            return SimpleNamespace(text='{"recovered": true}')

    class BusyModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("503 OVERLOADED: server busy")

    now = [0.0]
    first, second = RecoversOnSecondAttempt(), BusyModels()
    rw = _rewriter_with_models(first)
    rw.clients.append(SimpleNamespace(models=second))

    with patch("app.pipeline.gemini_rewriter.time.monotonic", side_effect=lambda: now[0]), \
         patch("app.pipeline.gemini_rewriter.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
        result = rw._generate("same request", ["model-a"])

    assert result == '{"recovered": true}'
    assert first.calls == 2
    assert second.calls == 1
    assert now[0] == 20.0


def test_daily_quota_exhaustion_immediately_uses_next_key_for_same_request():
    class DailyQuotaModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED daily quota per day exceeded")

    class WorkingModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(text='{"ok": true}')

    exhausted, working = DailyQuotaModels(), WorkingModels()
    rw = _rewriter_with_models(exhausted)
    rw.clients.append(SimpleNamespace(models=working))

    with patch("app.pipeline.gemini_rewriter.time.sleep") as sleep:
        result = rw._generate("same prompt", ["model-a"])
        fallback_result = rw._generate("next prompt", ["model-b"])

    assert result == '{"ok": true}'
    assert fallback_result == '{"ok": true}'
    assert exhausted.calls == 1
    assert working.calls == 2
    assert 0 in rw._exhausted
    sleep.assert_not_called()


def test_expired_story_context_deadline_stops_before_api_call():
    models = _RateLimitedModels()
    rw = _rewriter_with_models(models)
    rw._process_deadline = time.monotonic() - 1.0

    with pytest.raises(RuntimeError, match="reached the 20-minute limit"):
        rw._generate("prompt", ["model-a"])

    assert models.calls == 0


def test_gemini_http_client_uses_finite_timeout_and_no_hidden_retries():
    with patch("app.pipeline.gemini_rewriter.genai.Client") as client_factory:
        GeminiRewriter(api_key="test-key")

    options = client_factory.call_args.kwargs["http_options"]
    assert options.timeout == GeminiRewriter.HTTP_TIMEOUT_MS
    assert options.retry_options.attempts == 1


def test_story_context_reports_each_chunk_before_and_after_generation():
    class FakeRewriter:
        def _generate_json(self, prompt, model_order):
            return {
                "characters": [], "relationships": [], "places": [], "terms": [],
                "plot_state": [], "name_consistency": [], "unresolved": [],
            }

    units = [
        {"id": i, "timestamp": "00:00.00-00:01.00", "source_text": "x"}
        for i in range(1, 242)
    ]
    messages = []

    memory = build_story_context(
        FakeRewriter(), units, ["model"], lambda message, pct: messages.append((message, pct))
    )

    assert memory["characters"] == []
    assert any("1/2 ဖန်တီးနေပါသည်" in message for message, _ in messages)
    assert any("1/2 ပြီးပါပြီ" in message for message, _ in messages)
    assert any("2/2 ဖန်တီးနေပါသည်" in message for message, _ in messages)
    assert any("2/2 ပြီးပါပြီ" in message for message, _ in messages)


def test_recap_summarizes_large_transcript_chunks_and_keeps_source_time_ranges():
    class FakeRewriter:
        def __init__(self):
            self.prompts = []

        def _generate_json(self, prompt, model_order):
            self.prompts.append(prompt)
            index = len(self.prompts)
            return {"summary": f"Recap summary {index}.", "context_update": f"continuity-{index}"}

    source = [
        {"id": i, "start": float(i * 2), "end": float(i * 2 + 1), "text": "x"}
        for i in range(81)
    ]
    rw = FakeRewriter()
    context, summaries = run_recap_formula(rw, source, ["model-a"], "Burmese")

    assert len(rw.prompts) == 2
    assert "continuity-1" in rw.prompts[1]
    assert "55-70%" in rw.prompts[0]
    assert "do not collapse the chunk into only a few sentences" in rw.prompts[0]
    assert [item["text"] for item in summaries] == ["Recap summary 1.", "Recap summary 2."]
    assert [(item["start"], item["end"]) for item in summaries] == [(0.0, 159.0), (160.0, 161.0)]
    assert summaries[0]["source_segment_start_id"] == 0
    assert summaries[-1]["source_segment_end_id"] == 80
    assert all(item["is_summary_chunk"] for item in summaries)
    assert context["continuity"] == "continuity-2"


def test_recap_does_not_split_transcript_chunk_after_timeout():
    class FakeRewriter:
        def __init__(self):
            self.calls = 0

        def _generate_json(self, prompt, model_order):
            self.calls += 1
            raise RuntimeError("Gemini request timeout (504 DEADLINE_EXCEEDED)")

    source = [
        {"id": i, "start": float(i * 2), "end": float(i * 2 + 1), "text": f"event {i}"}
        for i in range(24)
    ]
    progress = []
    rw = FakeRewriter()

    with pytest.raises(RuntimeError, match="504 DEADLINE_EXCEEDED"):
        run_recap_formula(
            rw, source, ["model-a"], "Burmese", lambda message, pct: progress.append(message)
        )

    assert rw.calls == 1
    assert not any("ပိုသေးသော အပိုင်း" in message for message in progress)
