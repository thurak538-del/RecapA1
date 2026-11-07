import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from google import genai
from google.genai import types
from app.pipeline.burmese_text import normalize_myanmar_text
from app.pipeline.recap_formula import run_recap_formula

TARGET_LANGUAGE_NAMES = {
    "my": "Burmese (မြန်မာဘာသာ)",
    "en": "English",
    "th": "Thai (ထိုင်းဘာသာ)",
    "ja": "Japanese (ဂျပန်ဘာသာ)",
    "ko": "Korean (ကိုရီးယားဘာသာ)",
    "zh": "Chinese (Mandarin, Simplified)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "id": "Indonesian",
}


class GeminiRewriter:
    """Faithful, natural translation for timed subtitle segments."""

    DEFAULT_MODEL = "gemini-flash-latest"
    FALLBACK_MODEL = "gemini-3.8-flash"
    HTTP_TIMEOUT_MS = 60_000
    MAX_GENERATION_SECONDS = 180.0
    MAX_PROCESSING_SECONDS = 1200.0
    MAX_MODEL_CANDIDATES = 5
    MAX_TRANSIENT_RETRIES_PER_MODEL = 1
    FALLBACK_MODELS = [
        "gemini-flash-latest",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    _start_offset = 0   # shared across jobs

    def __init__(self, api_key, progress_callback: Optional[Callable[[str, float], None]] = None):
        """api_key: one key (str) or a list of keys that are rotated round-robin."""
        keys = [api_key] if isinstance(api_key, str) else list(api_key or [])
        keys = [k.strip() for k in dict.fromkeys(keys) if k and k.strip()]
        if not keys:
            raise ValueError("Gemini API Key is required. Please set it in Settings.")
        self.api_keys = keys
        retry_options = types.HttpRetryOptions(attempts=1)
        http_options = types.HttpOptions(timeout=self.HTTP_TIMEOUT_MS, retry_options=retry_options)
        self.clients = [genai.Client(api_key=k, http_options=http_options) for k in keys]
        self.client = self.clients[0]
        self.progress_callback = progress_callback
        GeminiRewriter._start_offset += 1
        self._cur = GeminiRewriter._start_offset % len(keys)   # next key to use (round-robin, spread across jobs)
        self._cooldown: Dict[Tuple[int, str], float] = {}   # (key, model) -> monotonic time it is usable again
        self._exhausted: set = set()       # key indices whose daily quota is used up for this job
        self._bad_keys: set = set()        # keys Google rejected as invalid
        self._process_deadline: Optional[float] = None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Parse model JSON, tolerating fences, stray control chars and trailing commas."""
        raw = (text or "").strip()
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        candidates = [cleaned]
        match = re.search(r"\{[\s\S]*\}", raw)
        if match and match.group(0) != cleaned:
            candidates.append(match.group(0))
        # Same candidates again with trailing commas removed ( ,} or ,] )
        candidates += [re.sub(r",\s*([}\]])", r"\1", c) for c in list(candidates)]

        last_exc: Optional[Exception] = None
        for cand in candidates:
            try:
                return json.loads(cand, strict=False)  # strict=False allows raw newlines/tabs in strings
            except Exception as exc:
                last_exc = exc
        if not match:
            raise ValueError("Model did not return a JSON object.")
        raise last_exc

    @staticmethod
    def _validate_segments(source: List[Dict[str, Any]], output: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(source) != len(output):
            return False, f"segment count changed: expected {len(source)}, got {len(output)}"
        for index, (src, dst) in enumerate(zip(source, output)):
            if dst.get("id") != src.get("id"):
                return False, f"segment {index} id changed"
            if abs(float(dst.get("start", -1)) - float(src.get("start", -2))) > 0.01:
                return False, f"segment {index} start timestamp changed"
            if abs(float(dst.get("end", -1)) - float(src.get("end", -2))) > 0.01:
                return False, f"segment {index} end timestamp changed"
            if not isinstance(dst.get("text"), str) or not dst["text"].strip():
                return False, f"segment {index} has empty text"
        return True, "ok"

    @staticmethod
    def _normalise_segments(source: List[Dict[str, Any]], output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep source timing/id authoritative even if the model formats numbers differently."""
        return [
            {
                "id": src["id"],
                "start": float(src["start"]),
                "end": float(src["end"]),
                "text": normalize_myanmar_text(str(dst.get("text", "")).strip()),
            }
            for src, dst in zip(source, output)
        ]

    @staticmethod
    def _normalise_hook(parsed: Dict[str, Any], source_segments: List[Dict[str, Any]], active_mode: str) -> Optional[Dict[str, Any]]:
        """Accept only a short, source-linked hook for Recap/Story modes."""
        if active_mode not in {"recap", "story"}:
            return None
        raw = parsed.get("hook") or {}
        if not isinstance(raw, dict) or raw.get("use_hook") is not True:
            return None
        text = normalize_myanmar_text(str(raw.get("text", "")).strip())
        source_ids = raw.get("source_segment_ids") or []
        valid_ids = {seg.get("id") for seg in source_segments}
        if not text or not isinstance(source_ids, list) or not source_ids:
            return None
        if not all(item in valid_ids for item in source_ids):
            return None
        # Recap only: the source opening may already be a deliberate hook. In
        # that case adding another generated hook would play two hooks back to
        # back. Story keeps its existing hook behavior because its opening is
        # authored as part of the narrative rewrite.
        if active_mode == "recap":
            opening_ids = {
                seg.get("id") for seg in source_segments
                if float(seg.get("start", 0.0) or 0.0) < 3.5
            }
            if source_ids and set(source_ids).issubset(opening_ids):
                return None
        # Keep the opening hook short enough to speak in roughly three
        # seconds; this is a guardrail in addition to the model instruction.
        if len(text) > 100 or len(source_ids) > 4:
            return None
        return {
            "id": "hook",
            "start": 0.0,
            "end": 0.0,
            "text": text,
            "role": "hook",
            "source_segment_ids": source_ids,
            "hook_type": str(raw.get("hook_type") or "conflict"),
            "confidence": float(raw.get("confidence") or 0.0),
        }

    @staticmethod
    def _retry_delay_seconds(error_text: str) -> Optional[float]:
        """Pull Google's suggested wait ("retry in 23.4s" / retryDelay: '23s') out of a 429 message."""
        m = re.search(r"retry in ([0-9.]+)\s*s", error_text) or re.search(r"retrydelay['\": ]+([0-9.]+)\s*s", error_text)
        return float(m.group(1)) if m else None

    def _pick_key(self, model: str) -> Tuple[Optional[int], Optional[float]]:
        """Return (key_index, 0) if a key is usable now, (None, seconds) if all are cooling down,
        or (None, None) if every key is daily-exhausted/invalid for this job and model."""
        n = len(self.clients)
        now = time.monotonic()
        shortest: Optional[float] = None
        for off in range(n):
            i = (self._cur + off) % n
            if i in self._bad_keys or i in self._exhausted:
                continue
            wait = self._cooldown.get((i, model), 0.0) - now
            if wait <= 0:
                return i, 0.0
            shortest = wait if shortest is None else min(shortest, wait)
        return None, shortest

    def _generate(self, prompt: str, model_candidates: List[str], temperature: float = 0.15) -> str:
        last_err = None
        n_keys = len(self.clients)
        candidates = list(dict.fromkeys(model_candidates or []))[:self.MAX_MODEL_CANDIDATES]
        started = time.monotonic()
        deadline = self._process_deadline or (started + self.MAX_GENERATION_SECONDS)
        while time.monotonic() < deadline:
            retry_cycle = False
            for candidate in candidates:
                attempts = 0
                max_attempts = max(1, n_keys)
                while attempts < max_attempts and time.monotonic() < deadline:
                    idx, wait = self._pick_key(candidate)
                    if idx is None:
                        if wait is None:
                            break  # all keys are daily-exhausted/invalid; try another model
                        last_err = RuntimeError(f"{candidate} is temporarily unavailable for {wait:.0f}s")
                        retry_cycle = True
                        break
                    attempts += 1
                    try:
                        response = self.clients[idx].models.generate_content(
                            model=candidate,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=temperature,
                                response_mime_type="application/json",
                            ),
                        )
                        if response.text:
                            self._cur = (idx + 1) % n_keys   # next request starts from the next key
                            return response.text
                        last_err = RuntimeError(f"{candidate} returned an empty response")
                        break
                    except Exception as exc:
                        last_err = exc
                        error_text = str(exc).lower()
                        if any(t in error_text for t in ("api key not valid", "api_key_invalid", "permission_denied", "leaked")):
                            print(f"[Gemini] key #{idx + 1} was rejected by Google; skipping it", flush=True)
                            self._bad_keys.add(idx)
                            self._cur = (idx + 1) % n_keys
                            continue
                        if "429" in error_text or "resource_exhausted" in error_text or "quota" in error_text:
                            next_idx = (idx + 1) % n_keys
                            if "perday" in error_text or "per day" in error_text:
                                print(f"[Gemini] key #{idx + 1} / {candidate}: daily quota used; skipping this key for the rest of the job and immediately switching to key #{next_idx + 1}", flush=True)
                                self._exhausted.add(idx)
                            else:
                                delay = self._retry_delay_seconds(error_text) or 30.0
                                self._cooldown[(idx, candidate)] = time.monotonic() + min(delay + 1.5, 75.0)
                                retry_cycle = True
                                print(f"[Gemini] key #{idx + 1} / {candidate}: rate limit; immediately switching to key #{next_idx + 1}", flush=True)
                            self._cur = next_idx
                            continue
                        transient = any(marker in error_text for marker in (
                            "503", "500", "temporarily unavailable", "overloaded", "unavailable",
                            "busy", "deadline_exceeded", "deadline expired", "gateway timeout",
                            "504", "timed out", "timeout",
                        ))
                        if transient:
                            next_idx = (idx + 1) % n_keys
                            self._cooldown[(idx, candidate)] = time.monotonic() + 20.0
                            self._cur = next_idx
                            retry_cycle = True
                            print(
                                f"[Gemini] key #{idx + 1} / {candidate}: busy or deadline timeout; "
                                f"immediately retrying the same request with key #{next_idx + 1}",
                                flush=True,
                            )
                            continue
                        break
            if time.monotonic() >= deadline or not retry_cycle:
                break
            now = time.monotonic()
            waits = [
                available_at - now
                for (key_index, model), available_at in self._cooldown.items()
                if model in candidates and key_index not in self._bad_keys
                and key_index not in self._exhausted and available_at > now
            ]
            if not waits:
                break
            remaining = max(0.0, deadline - now)
            sleep_for = min(min(waits), remaining)
            if sleep_for <= 0.0:
                break
            print(f"[Gemini] all usable keys are temporarily busy; rotating again in {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
        elapsed = time.monotonic() - started
        if self._process_deadline is not None and time.monotonic() >= self._process_deadline:
            last_err = TimeoutError(f"Gemini context/story processing reached the {self.MAX_PROCESSING_SECONDS / 60:.0f}-minute limit")
        raise RuntimeError(
            f"Gemini API processing stopped after {elapsed:.1f}s; tried {len(candidates)} model candidate(s). "
            f"Last error: {last_err}. Check API quota/key availability and retry."
        )

    def _generate_json(self, prompt: str, model_candidates: List[str], attempts: int = 2) -> Dict[str, Any]:
        """Generate and parse JSON; if the model returns broken JSON, ask again (slightly higher temperature)."""
        last_exc: Optional[Exception] = None
        for n in range(attempts):
            text = self._generate(prompt, model_candidates, temperature=min(0.15 + 0.25 * n, 0.7))
            try:
                return self._extract_json(text)
            except Exception as exc:
                last_exc = exc
                print(f"[Gemini] invalid JSON from model ({exc}); re-asking {n + 1}/{attempts}", flush=True)
        raise RuntimeError(f"Gemini returned invalid JSON {attempts} times: {last_exc}")

    def _available_model_candidates(self) -> List[str]:
        """Return known fallbacks without an extra models.list() network round-trip."""
        return list(dict.fromkeys([self.DEFAULT_MODEL, *self.FALLBACK_MODELS]))[:self.MAX_MODEL_CANDIDATES]

    def process(
        self,
        groq_result: Dict[str, Any],
        output_dir: Path,
        mode: str = "translate",
        target_language: str = "my",
        model_name: str = DEFAULT_MODEL,
        source_duration: float = 0.0,
        processing_mode: str = "recap",
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._process_deadline = time.monotonic() + self.MAX_PROCESSING_SECONDS
        source_segments = groq_result.get("segments", [])
        if not source_segments:
            raise ValueError("Transcript has no timed segments to translate.")

        lang_name = TARGET_LANGUAGE_NAMES.get(target_language, target_language)
        source_text = groq_result.get("text", "")
        duration_rule = ""
        active_mode = str(processing_mode or "recap").lower()
        if source_duration > 0 and active_mode == "story":
            duration_rule = f"""
STORY COVERAGE AND LENGTH RULE:
The source video is approximately {source_duration:.1f} seconds long. This is a
full storytelling rewrite, not a short summary, but it must not repeat the source
duration minute-for-minute. Cover the important sequence of events, actions,
decisions, emotions, dangers, reveals, and consequences from the source. Do not
collapse a long video into a tiny recap, but also do not narrate every pause or
repeated visual detail. Target a natural spoken script of roughly 60–70% of the
source duration when the source is long (for example, a 9-minute video should
normally become about 5–6 minutes). Preserve the story's key meaning and suspense.
The renderer will adjust the video to the final TTS duration.
"""
        elif source_duration > 0 and active_mode == "dubbing":
            duration_rule = f"""
DUBBING TIMING RULE:
The source dialogue is approximately {source_duration:.1f} seconds long. Keep the
translation close to the original timing. Do not intentionally expand the dialogue,
repeat content, or pad it to reach one minute.
"""
        elif source_duration > 0:
            duration_rule = f"""
NATURAL LENGTH RULE:
The source narration is approximately {source_duration:.1f} seconds long. Make the
spoken translation only slightly longer than the source, normally by about 10 to
15 seconds and never intentionally more than about 20 seconds longer.
Use natural connective wording only when it clarifies an event already present in
the source. Do not pad the narration to reach one minute, do not target 61 seconds,
and do not repeat a sentence, event, or conclusion to increase the duration.
Keep the original segment ids and timestamps; put the naturally expanded wording
into the corresponding segment texts.
"""
        if self.progress_callback:
            message = (
                "Transcript ကို အပိုင်းလိုက် အကျဉ်းချုပ်နေပါသည်..."
                if active_mode == "recap"
                else "Transcript အပြည့်ကို ဖတ်ပြီး video အမျိုးအစား ခွဲနေပါသည်..."
            )
            self.progress_callback(message, 10.0)

        # RECAP MODE: summarize chronological chunks once instead of a separate
        # context pass, segment translation pass, and audit pass.
        if active_mode == "recap":
            model_order = self._available_model_candidates()
            context, formula_segments = run_recap_formula(
                self, source_segments, model_order, lang_name, self.progress_callback
            )
            processed_segments = [
                {**segment, "text": normalize_myanmar_text(str(segment.get("text", "")).strip())}
                for segment in formula_segments
            ]
            processed_full_text = "\n".join(s["text"] for s in processed_segments)
            metadata = {
                "content_type": "recap",
                "tone": f"concise chunk-wise narrator recap ({lang_name})",
                "glossary": [],
                "story_context": context,
                "source_segment_count": len(source_segments),
                "summary_chunk_count": len(processed_segments),
                "meaning_policy": "chunkwise_contextual_summary_recap",
                "model_order": model_order,
                "hook": None,
            }
            (output_dir / "translation_analysis.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            txt_path = output_dir / "processed_transcript.txt"
            txt_path.write_text(
                processed_full_text + "\n\n--- PROCESSED SEGMENTS WITH TIMESTAMPS ---\n" +
                "".join(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}\n" for s in processed_segments),
                encoding="utf-8")
            json_path = output_dir / "processed_transcript.json"
            json_path.write_text(
                json.dumps({**metadata, "full_text": processed_full_text, "segments": processed_segments,
                            "target_language": target_language, "mode": mode}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            if self.progress_callback:
                self.progress_callback("အပိုင်းလိုက် အကျဉ်းချုပ် recap ပြီးပါပြီ။", 100.0)
            return {"txt_path": txt_path, "json_path": json_path, "full_text": processed_full_text,
                    "segments": processed_segments, "hook": None, "analysis": metadata}

        story_mode_prompt = """
You are a professional Burmese movie, drama, and anime recap storyteller.

Transform a transcript that may contain conversations between two or more movie
characters into a clear, engaging Burmese story narration. This is a narrator
recap, not a line-by-line dubbing script.

MULTI-CHARACTER STORY RULES:
- First understand each conversation in its scene context: who is speaking,
  what they want, what they know, what they hide, and how the exchange changes
  the story.
- Do not translate every dialogue line separately. Convert dialogue into smooth
  narrator sentences that explain the important meaning and consequence.
- Keep character identities and relationships clear. Use a name or a short
  description when the audience could otherwise confuse two characters.
- Preserve important arguments, promises, threats, discoveries, decisions,
  betrayals, emotional changes, and information revealed through dialogue.
- Never invent a speaker, motive, relationship, event, or twist that is not
  supported by the transcript.
- Do not output speaker labels, quotation-heavy dialogue, bullet points, scene
  headings, or chapter labels. Write one continuous story flow for TTS.

STORY COVERAGE:
- This is a complete movie recap, not a shallow short summary.
- Follow the main chain of cause and effect from setup to consequence.
- Include important visual actions as well as important conversations.
- Remove only repeated greetings, filler reactions, long pauses, and duplicate
  information that do not change the plot.
- Do not repeat an event merely to make the narration longer.

WRITING STYLE:
- Natural spoken Burmese with short, easy-to-hear sentences.
- Explain the scene like a skilled storyteller speaking to a friend.
- Focus on tension, emotion, danger, clever decisions, mistakes, reversals,
  and consequences without exaggerated clickbait.
- Use natural transitions so the audience understands why the next event occurs.
- Build curiosity from the actual situation before an important reveal.
- Avoid robotic phrasing and repetitive openings such as “ဒါပေမယ့် သူ မသိသေးတာက...”
  or “နောက်ထပ် ဖြစ်လာမယ့်အရာက...”.

OUTPUT:
Generate a complete Burmese movie-story recap optimized for YouTube, TikTok,
and Facebook narration. Return only the requested JSON structure; the final
spoken text must remain a continuous narrator script.
"""
        educational_rewrite_prompt = """
You are an expert Burmese educational storyteller and translator.

Transform the source narration into natural, engaging Burmese while preserving all original facts and meaning.

IMPORTANT:
- Do NOT summarize.
- Do NOT skip information.
- Do NOT change the meaning.
- Do NOT invent facts.
- Keep all important details from the original.

GOAL:
Rewrite the narration so it feels more interesting, easier to follow, and more engaging than a direct translation. Create a curious, educational documentary feeling without clickbait.

STYLE:
- Natural spoken Burmese.
- Educational documentary style.
- Curious and engaging tone.
- Slightly conversational, clear, and easy to understand.
- Sound like a human narrator.

RETENTION RULES:
- Add smooth transitions between ideas.
- Briefly create curiosity before explaining a new concept.
- Highlight surprising facts naturally.
- Emphasize cause-and-effect relationships.
- Make technical explanations easier to visualize.
- Slightly expand explanations only when needed for clarity.

LENGTH:
- The Burmese version may be up to 10-20% longer than the original when needed for clarity and flow.
- Never make it shorter by removing information.

HOOK RULE:
- The first 1-3 sentences should create immediate curiosity about the topic.
- Use a source-grounded hook, not clickbait.
- Do not invent a mystery, fact, event, or ending that is absent from the source.
"""
        mode_instructions = {
            "recap": educational_rewrite_prompt,
            "story": story_mode_prompt,
            "dubbing": "Translate as natural spoken dubbing for the original scene. Preserve each speaker's meaning, emotion, intensity, and timing; do not summarize or turn dialogue into a narrator recap.",
        }.get(str(processing_mode or "recap").lower(), "Rewrite as a natural movie-recap narration.")
        prompt = f"""
Translate the transcript into natural spoken {lang_name}.
MODE: {processing_mode}
MODE INSTRUCTION: {mode_instructions}

Rules:

* Translate the MEANING, not word-for-word.
* Make it sound like a native speaker naturally telling a story, NOT like a book or machine translation.
* Do not copy the original sentence structure if it sounds unnatural.
* Do not summarize or remove important information.
* Do not add explanations, details, opinions, or information that is not in the original.
* Follow the mode-specific timing and coverage rule below; do not pad or repeat content.
* Use natural conversational grammar and expressions.
* Avoid overly formal/literary language.
* Avoid unnecessary pronouns and words such as “၎င်း”, “၎င်းတို့”, “ဖြစ်သည်”, “ဖြစ်ကြသည်” in Burmese when they make the sentence sound unnatural.
* Preserve names, numbers, actions, events, and important details accurately.
* Make every sentence smooth and easy to understand when heard through TTS.
* Think like a native speaker explaining what happened in a movie to a friend.
* Translate the story, not the words.
{duration_rule}

SOURCE-COVERAGE SAFETY:
* Translate every sentence and every proposition in every source segment.
* The output for each segment must carry the complete meaning of its corresponding source segment.
* Do not remove details, qualifiers, uncertainty, repetition that carries emphasis, or emotional meaning.
* Keep every segment. Do not merge, split, reorder, or invent segments.
* Keep every input id, start, and end EXACTLY unchanged.
* Use the full transcript only to resolve pronouns or context; never use it to summarize segments.

OUTPUT: Return ONLY valid JSON with this exact shape:
{{
  "content_type": "entertainment|educational|emotional_story|news_documentary|conversation",
  "tone": "short description",
  "glossary": [{{"source": "term", "target": "translation"}}],
  "full_text": "joined translated text",
  "hook": {{
    "use_hook": true,
    "text": "1 to 3 short source-grounded Burmese sentences",
    "source_segment_ids": [0],
    "hook_type": "danger|mystery|conflict|unexpected_change|emotion|none",
    "confidence": 0.0
  }},
  "segments": [
    {{"id": 0, "start": 0.0, "end": 2.5, "text": "translation"}}
  ]
}}

SOURCE FULL TRANSCRIPT:
{source_text}

SOURCE TIMED SEGMENTS:
{json.dumps(source_segments, ensure_ascii=False, indent=2)}

HOOK RULES (Recap and Story only):
* First inspect the full transcript and choose the strongest source-supported conflict,
  danger, mystery, unexpected change, or turning point.
* Write one very short natural Burmese opening hook from that event, without inventing facts.
* Recap mode only: if the source video already begins with a clear hook or
  attention-grabbing opening, return use_hook=false so the original hook is not
  followed by a duplicate generated hook. Do not apply this special rule to Story mode.
* Do not reveal the complete ending, identity reveal, or full twist.
* The hook is a preview, not a new event and not a summary of the whole video.
* Link it to the exact source segment ids used.
* Target approximately 3 seconds when spoken: one short sentence or two very short clauses,
  normally no more than about 100 Unicode characters. If it cannot fit naturally, return
  use_hook=false and an empty text.
* If no clear supportable hook exists, return use_hook=false and an empty text.
* For Dubbing mode, always return use_hook=false; translate original dialogue only.
* The main segments must still preserve every source segment and must not be removed.
"""

        if self.progress_callback:
            self.progress_callback("Context-aware ဘာသာပြန်နေပါသည်... (meaning ကို ထိန်းထားပါသည်)", 35.0)
        # Try all known Flash fallbacks, then models exposed by this API key.
        model_order = self._available_model_candidates()
        response_text = self._generate(prompt, model_order)
        try:
            parsed = self._extract_json(response_text)
            translated = parsed.get("segments", [])
            valid, reason = self._validate_segments(source_segments, translated)
        except Exception as exc:
            valid, reason, parsed = False, str(exc), {}

        if not valid:
            if self.progress_callback:
                self.progress_callback("ဘာသာပြန်ရလဒ်ကို မူရင်း timestamp နဲ့ ပြန်စစ်နေပါသည်...", 62.0)
            repair_prompt = f"""
Repair this translation JSON without rewriting the wording.
Return ONLY JSON with content_type, tone, glossary, full_text, and segments.
The segments list MUST have exactly the same count as SOURCE, and each id/start/end
MUST be copied exactly from SOURCE. Only fix missing/invalid structure; preserve text.
SOURCE={json.dumps(source_segments, ensure_ascii=False)}
BAD_RESULT={json.dumps(parsed, ensure_ascii=False)}
VALIDATION_ERROR={reason}
"""
            repaired = self._extract_json(self._generate(repair_prompt, model_order))
            translated = repaired.get("segments", [])
            valid, reason = self._validate_segments(source_segments, translated)
            parsed = repaired

        if not valid:
            raise RuntimeError(f"Translation validation failed: {reason}")

        processed_segments = self._normalise_segments(source_segments, translated)
        hook = self._normalise_hook(parsed, source_segments, active_mode)
        if hook:
            processed_segments = [hook, *processed_segments]
        processed_full_text = "\n".join(s["text"] for s in processed_segments)
        metadata = {
            "content_type": parsed.get("content_type", "conversation"),
            "tone": parsed.get("tone", "natural and faithful"),
            "glossary": parsed.get("glossary", []),
            "source_segment_count": len(source_segments),
            "meaning_policy": "faithful_natural_translation_no_summarization",
            "model_order": model_order,
            "hook": hook,
        }

        if self.progress_callback:
            self.progress_callback("ဘာသာပြန်ရလဒ်ကို အဓိပ္ပာယ်/အရှည် စစ်ဆေးနေပါသည်...", 85.0)

        (output_dir / "translation_analysis.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        processed_txt_path = output_dir / "processed_transcript.txt"
        processed_txt_path.write_text(
            processed_full_text + "\n\n--- PROCESSED SEGMENTS WITH TIMESTAMPS ---\n" +
            "".join(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}\n" for s in processed_segments),
            encoding="utf-8",
        )
        processed_json_path = output_dir / "processed_transcript.json"
        processed_json_path.write_text(
            json.dumps({**metadata, "full_text": processed_full_text, "segments": processed_segments,
                        "target_language": target_language, "mode": mode}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.progress_callback:
            self.progress_callback("Controlled natural ဘာသာပြန်ပြီးပါပြီ။", 100.0)
        return {
            "txt_path": processed_txt_path,
            "json_path": processed_json_path,
            "full_text": processed_full_text,
            "segments": processed_segments,
            "hook": hook,
            "analysis": metadata,
        }
