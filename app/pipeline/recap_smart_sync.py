"""Recap segmentation and Smart Sync with preserved silent intervals.

Ordinary speech picture intervals are aligned to normal-speed TTS. Summary
segments sample representative shots across their source range and never speed
up the whole summarized chunk. Intro, gaps, and trailing non-speech picture are
retained and accelerated at most 1.1x with matching narration/background timing.
"""
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.pipeline.gpu_utils import get_video_encoder_args, get_active_encoder_name

MAX_VIDEO_SPEED = 1.25
MAX_SILENT_VIDEO_SPEED = 1.1
FPS = 30

_ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "no.", "mt.",
           "gen.", "col.", "lt.", "capt.", "sgt.", "u.s.", "u.k.", "a.m.", "p.m."}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _comparison_text(value: Any) -> str:
    return re.sub(r"[^\w\u1000-\u109f]+", "", _clean(value).lower(), flags=re.UNICODE)


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _word_ends_sentence(text: str) -> bool:
    cleaned = _clean(text).lower()
    if cleaned in _ABBREV:
        return False
    if re.fullmatch(r"[a-z]\.['’\"”)]?", cleaned):
        return False
    return bool(re.search(r"[.!?…။।॥。！？][\"'’”)]?$", cleaned))


def _word_ends_soft_clause(text: str) -> bool:
    return bool(re.search(r"[;,:၊；，、][\"'’”)]?$", _clean(text)))


def _fallback_tokens(text: str, start: float, end: float, tid: int) -> List[Dict[str, Any]]:
    words = re.findall(r"\S+", _clean(text))
    if not words or end <= start:
        return []
    step = (end - start) / len(words)
    return [{"text": w, "start": start + i * step, "end": start + (i + 1) * step,
             "source_transcript_id": tid} for i, w in enumerate(words)]


def _word_tokens(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Measured word times (transcript['words']) when present, else even split per segment."""
    measured: Dict[int, List[Dict[str, Any]]] = {}
    for w in transcript.get("words") or []:
        text, st = _clean(w.get("text")), float(w.get("start", 0.0) or 0.0)
        en = float(w.get("end", st) or st)
        if text and en > st:
            measured.setdefault(int(w.get("source_transcript_id", -1)), []).append(
                {"text": text, "start": st, "end": en, "source_transcript_id": int(w.get("source_transcript_id", -1))})
    tokens: List[Dict[str, Any]] = []
    for idx, seg in enumerate(transcript.get("segments") or []):
        text = _clean(seg.get("text"))
        st = float(seg.get("start", 0.0) or 0.0)
        en = float(seg.get("end", st) or st)
        if not text or en <= st:
            continue
        tid = int(seg.get("id", idx))
        tokens.extend(measured.get(tid) or _fallback_tokens(text, st, en, tid))
    tokens.sort(key=lambda t: (float(t["start"]), float(t["end"])))
    return tokens


def atomicize_units(transcript: Dict[str, Any], *, max_duration: float = 5.6,
                    target_duration: float = 4.2) -> List[Dict[str, Any]]:
    tokens = _word_tokens(transcript)
    units: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    target_duration = max(2.5, float(target_duration))
    max_duration = max(target_duration + 0.5, float(max_duration))
    min_natural = min(1.8, target_duration * 0.4)

    def close(boundary: str) -> None:
        nonlocal current
        if not current:
            return
        ids = list(dict.fromkeys(int(w["source_transcript_id"]) for w in current))
        units.append({
            "source_transcript_ids": ids,
            "speech_start": round(float(current[0]["start"]), 3),
            "speech_end": round(float(current[-1]["end"]), 3),
            "source_text": _clean(" ".join(str(w["text"]) for w in current)),
            "semantic_boundary": boundary,
        })
        current = []

    for i, tok in enumerate(tokens):
        current.append(tok)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        gap = 0.0
        if i + 1 < len(tokens):
            gap = max(0.0, float(tokens[i + 1]["start"]) - float(tok["end"]))
        if gap >= 1.0:
            close("pause")
        elif _word_ends_sentence(tok["text"]):
            close("sentence")
        elif _word_ends_soft_clause(tok["text"]) and duration >= min_natural:
            close("clause")
        elif duration >= min_natural and gap >= 0.50:
            close("pause")
        elif duration >= target_duration and gap >= 0.16:
            close("pause")
        elif duration >= max_duration:
            close("forced_local")
    close("end_of_transcript")

    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(units):
        u = units[i]
        dur = float(u["speech_end"]) - float(u["speech_start"])
        ntok = len(re.findall(r"\S+", u["source_text"]))
        nxt_gap = (float(units[i + 1]["speech_start"]) - float(u["speech_end"])) if i + 1 < len(units) else 9999.0
        if dur <= 0.5 and ntok <= 2 and i + 1 < len(units) and nxt_gap <= 0.60:
            f = dict(units[i + 1])
            lt, rt = _clean(u["source_text"]), _clean(f["source_text"])
            lw = _comparison_text(lt.rstrip(".!?…。！？"))
            rf = _comparison_text(rt.split(maxsplit=1)[0] if rt else "")
            f["speech_start"] = u["speech_start"]
            f["source_text"] = rt if lw and lw == rf else _clean(f"{lt} {rt}")
            f["source_transcript_ids"] = list(dict.fromkeys(list(u["source_transcript_ids"]) + list(f["source_transcript_ids"])))
            units[i + 1] = f
            i += 1
            continue
        merged.append(u)
        i += 1
    for n, u in enumerate(merged, 1):
        u["id"] = n
        u["timestamp"] = f"{_fmt_ts(u['speech_start'])} --> {_fmt_ts(u['speech_end'])}"
    return merged


def assign_full_coverage_boundaries(units: List[Dict[str, Any]], video_duration: float) -> List[Dict[str, Any]]:
    """Give each spoken unit the video from its source speech start to the next.

    Previous midpoint boundaries included part of the next utterance's lead-in
    in the preceding visual slot. That made the new TTS sentence begin before
    the corresponding visual action. Cutting at the next source speech start
    keeps every boundary contiguous while anchoring each unit to its speech.
    Any pre-speech intro remains a separate lead-in interval for the caller.
    """
    if not units:
        return []
    out = [dict(x) for x in units]
    duration = max(0.0, float(video_duration))
    starts = []
    previous = 0.0
    for unit in out:
        try:
            start = float(unit.get("speech_start", 0.0))
        except (TypeError, ValueError):
            start = previous
        start = max(previous, min(duration, max(0.0, start)))
        starts.append(start)
        previous = start
    out[0]["video_start"] = round(starts[0], 3)
    for i in range(len(out) - 1):
        boundary = max(starts[i], starts[i + 1])
        out[i]["video_end"] = round(boundary, 3)
        out[i + 1]["video_start"] = round(boundary, 3)
    out[-1]["video_end"] = round(duration, 3)
    return out


def sync_filter(source_duration: float, target_duration: float, max_speed: float = MAX_VIDEO_SPEED) -> str:
    """Exact setpts to the narration slot; speed-up capped at max_speed, then trimmed."""
    desired = target_duration / source_duration
    min_factor = 1.0 / max(1.0, max_speed)
    if desired < min_factor:
        return (f"trim=duration={source_duration:.6f},setpts={min_factor:.9f}*(PTS-STARTPTS),"
                f"trim=duration={target_duration:.6f},setpts=PTS-STARTPTS,fps={FPS}")
    return f"trim=duration={source_duration:.6f},setpts={desired:.9f}*(PTS-STARTPTS),fps={FPS}"


_LONG_EDGE = {"tiktok540": 960, "tiktok720": 1280, "1080p": 1920, "tiktok1080": 1920, "2k": 2560, "tiktok2k": 2560, "4k": 3840, "tiktok4k": 3840}


def _probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", str(path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _atempo_chain(speed: float) -> str:
    """Build an FFmpeg atempo chain for any positive audio speed ratio."""
    speed = max(0.01, float(speed))
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def _summary_visual_clips(source_start: float, source_end: float, target_duration: float,
                          *, max_shot_duration: float = 2.5) -> List[Dict[str, float]]:
    """Pick evenly spaced source shots whose total playback is near real time.

    Summary narration may cover many minutes of source material in a few seconds.
    Selecting short shots across that range avoids the old extreme setpts speed-up
    while still showing representative moments throughout the summarized chunk.
    """
    span = max(0.0, float(source_end) - float(source_start))
    target = max(0.0, float(target_duration))
    if span <= 0.0 or target <= 0.0:
        return []
    playback = min(span, target)
    count = max(1, int(math.ceil(playback / max(0.25, max_shot_duration))))
    shot = playback / count
    spacing = span / count
    clips = []
    for index in range(count):
        center = source_start + (index + 0.5) * spacing
        start = min(max(source_start, center - shot / 2.0), source_end - shot)
        clips.append({"start": round(start, 6), "end": round(start + shot, 6)})
    return clips


def build_mouth_sync_plan(
    segments: List[Dict[str, Any]],
    video_duration: float,
    *,
    fps: int = FPS,
    silent_speed: float = MAX_SILENT_VIDEO_SPEED,
) -> List[Dict[str, Any]]:
    """Fit speech pictures to TTS and preserve silence footage at up to 1.1x.

    Narration excerpts are copied at their original speed. Every source silent
    interval in the intro, gaps, and ending remains represented in the output;
    none is trimmed. Those intervals are accelerated by silent_speed (clamped to
    1.0x–1.1x). The matching narration
    silence is shortened by the same ratio. A frame hold is added only if an
    original inter-line gap is shorter than the TTS engine's natural pause.
    """
    if not segments:
        return []
    duration = max(0.0, float(video_duration))
    fps = max(1, int(fps))
    silent_speed = min(MAX_SILENT_VIDEO_SPEED, max(1.0, float(silent_speed)))
    frame_count = max(1, int(round(duration * fps)))
    items = []
    previous_end_frame = 0
    for segment in segments:
        try:
            source_start = max(0.0, min(duration, float(segment["source_start"])))
            source_end = max(source_start, min(duration, float(segment["source_end"])))
            audio_start = float(segment["start"])
            audio_end = float(segment["end"])
            slot_duration = max(0.0, float(segment.get("slot_duration", audio_end - audio_start)))
        except (KeyError, TypeError, ValueError):
            continue
        if audio_end <= audio_start or source_start >= duration or previous_end_frame >= frame_count:
            continue
        start_frame = max(previous_end_frame, int(round(source_start * fps)))
        start_frame = min(frame_count - 1, max(0, start_frame))
        end_frame = min(frame_count, max(start_frame + 1, int(round(source_end * fps))))
        previous_end_frame = end_frame
        items.append({
            "segment": segment,
            "video_start": start_frame / float(fps),
            "video_end": min(duration, end_frame / float(fps)),
            "audio_start": audio_start,
            "audio_end": audio_end,
            "speech_duration": audio_end - audio_start,
            "pause_duration": max(0.0, slot_duration - (audio_end - audio_start)),
        })
    if not items:
        return []

    plan: List[Dict[str, Any]] = []
    cursor = 0.0

    def add_silence(video_start: float, video_end: float, *, hold: bool = False,
                    output_duration: Optional[float] = None) -> float:
        nonlocal cursor
        source_gap = max(0.0, video_end - video_start)
        target = source_gap / silent_speed if output_duration is None else max(0.0, output_duration)
        if output_duration is None:
            target = math.ceil(target * fps - 1e-9) / float(fps)
        if target <= 0.001:
            return 0.0
        plan.append({
            "video_start": video_start,
            "video_end": video_end,
            "slot_duration": target,
            "audio_type": "silence",
            "hold_frame": hold,
        })
        cursor += target
        return target

    add_silence(0.0, items[0]["video_start"])
    for index, item in enumerate(items):
        speech_duration = max(0.04, item["speech_duration"])
        segment = item["segment"]
        plan.append({
            "video_start": item["video_start"],
            "video_end": item["video_end"],
            "slot_duration": speech_duration,
            "audio_type": "speech",
            "audio_start": item["audio_start"],
            "audio_end": item["audio_end"],
        })
        if segment.get("is_summary_chunk"):
            clips = _summary_visual_clips(
                item["video_start"], item["video_end"], speech_duration
            )
            if clips:
                plan[-1]["video_clips"] = clips
        segment["start"] = round(cursor, 6)
        segment["slot_start"] = round(cursor, 6)
        segment["end"] = round(cursor + speech_duration, 6)
        cursor += speech_duration

        pause_duration = item["pause_duration"]
        if index + 1 < len(items):
            gap_start = item["video_end"]
            gap_end = max(gap_start, items[index + 1]["video_start"])
            preserved_gap = add_silence(gap_start, gap_end)
            extra_pause = max(0.0, pause_duration - preserved_gap)
            if extra_pause > 0.001:
                hold_start = max(0.0, item["video_end"] - 1.0 / fps)
                add_silence(hold_start, item["video_end"], hold=True,
                            output_duration=extra_pause)
            segment["slot_duration"] = round(speech_duration + max(pause_duration, preserved_gap), 6)
        else:
            segment["slot_duration"] = round(speech_duration, 6)
            add_silence(item["video_end"], duration)
    return plan


def build_source_timed_narration(
    tts_audio_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Reassemble unchanged-speed TTS excerpts with plan-matched silent intervals."""
    tts_audio_path, output_path = Path(tts_audio_path), Path(output_path)
    filters: List[str] = []
    labels: List[str] = []
    planned_duration = 0.0
    for index, segment in enumerate(segments):
        duration = max(0.0, float(segment.get("slot_duration", 0.0) or 0.0))
        if duration <= 0.001:
            continue
        planned_duration += duration
        label = f"part{index}"
        labels.append(f"[{label}]")
        if segment.get("audio_type") == "speech":
            audio_start = max(0.0, float(segment["audio_start"]))
            audio_end = float(segment["audio_end"])
            if audio_end <= audio_start:
                raise ValueError(f"Invalid TTS excerpt for speech segment {index}.")
            filters.append(
                f"[0:a]atrim=start={audio_start:.6f}:end={audio_end:.6f},"
                f"asetpts=PTS-STARTPTS,aresample=44100,"
                f"aformat=sample_rates=44100:channel_layouts=mono,"
                f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}[{label}]"
            )
        else:
            filters.append(
                f"anullsrc=r=44100:cl=mono,atrim=duration={duration:.6f},"
                f"asetpts=PTS-STARTPTS[{label}]"
            )
    if not labels:
        raise ValueError("No speech or silence segments were provided for Smart Sync audio.")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[out]")
    else:
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    aligned_duration = math.ceil(planned_duration * FPS - 1e-9) / float(FPS)
    filters.append(
        f"[out]apad=whole_dur={aligned_duration:.6f},"
        f"atrim=duration={aligned_duration:.6f}[aligned]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tts_audio_path),
         "-filter_complex", ";".join(filters), "-map", "[aligned]", "-c:a", "pcm_s16le",
         "-ar", "44100", "-ac", "1", str(output_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Timed narration assembly failed: {(result.stderr or '')[-1500:]}")
    return output_path


def render_smart_sync(
    video_path: Path,
    tts_audio_path: Path,
    segments: List[Dict[str, Any]],
    output_path: Path,
    work_dir: Path,
    resolution: str = "1080p",
    enable_4k_filter: bool = False,
    mirror_mode_7s: bool = False,
    progress: Optional[Callable[[str, float], None]] = None,
    background_audio_path: Optional[Path] = None,
) -> Path:
    """segments: [{video_start, video_end, slot_duration}] in order (slot = narration time incl. pause)."""
    video_path, tts_audio_path, output_path = Path(video_path), Path(tts_audio_path), Path(output_path)
    work_dir = Path(work_dir)
    parts_dir = work_dir / "smart_sync_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("*.mp4"):
        old.unlink(missing_ok=True)

    edge = _LONG_EDGE.get(str(resolution or "1080p").strip().lower(), 1920)
    scale = (f"scale=w='if(gte(iw,ih),{edge},-2)':h='if(gte(iw,ih),-2,{edge})':flags=lanczos")
    encoder_args = get_video_encoder_args(cq=18, crf=17)
    cpu_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p"]
    use_cpu = False

    parts: List[Path] = []
    cursor, frames_done, total = 0.0, 0, len(segments)
    for idx, seg in enumerate(segments, 1):
        src_dur = max(0.04, float(seg["video_end"]) - float(seg["video_start"]))
        target = max(0.04, float(seg["slot_duration"]))
        cursor += target
        frames = max(1, int(math.ceil(cursor * FPS - 1e-9)) - frames_done)
        frames_done += frames
        video_clips = seg.get("video_clips") or []
        if video_clips:
            clip_filters = []
            clip_labels = []
            clip_playback = 0.0
            for clip_index, clip in enumerate(video_clips):
                clip_start = max(0.0, float(clip["start"]))
                clip_end = max(clip_start + 0.04, float(clip["end"]))
                clip_playback += clip_end - clip_start
                label = f"visual{idx}_{clip_index}"
                clip_labels.append(f"[{label}]")
                clip_filters.append(
                    f"[0:v]trim=start={clip_start:.6f}:end={clip_end:.6f},"
                    f"setpts=PTS-STARTPTS[{label}]"
                )
            clip_filters.append(
                f"{''.join(clip_labels)}concat=n={len(clip_labels)}:v=1:a=0,"
                f"fps={FPS},{scale},tpad=stop_mode=clone:"
                f"stop_duration={max(0.5, target - clip_playback + 0.05):.6f}[outv]"
            )
            summary_filter = ";".join(clip_filters)
            vf = None
        else:
            vf = f"{sync_filter(src_dur, target, max_speed=1000.0)},{scale},tpad=stop_mode=clone:stop_duration=0.5"
            summary_filter = None
        dst = parts_dir / f"part_{idx:05d}.mp4"

        def _encode(enc_args: List[str]) -> subprocess.CompletedProcess:
            if summary_filter:
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                       "-i", str(video_path), "-filter_complex", summary_filter,
                       "-map", "[outv]", "-an", "-frames:v", str(frames),
                       *enc_args, "-movflags", "+faststart", str(dst)]
            else:
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                       "-ss", f"{float(seg['video_start']):.6f}", "-t", f"{src_dur:.6f}",
                       "-i", str(video_path), "-an", "-vf", vf, "-frames:v", str(frames),
                       *enc_args, "-movflags", "+faststart", str(dst)]
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        res = _encode(cpu_args if use_cpu else encoder_args)
        if res.returncode != 0 and not use_cpu and not parts:
            # GPU encoder unusable: switch the whole job to CPU (first part only, so codecs never mix)
            use_cpu = True
            res = _encode(cpu_args)
        if res.returncode != 0:
            raise RuntimeError(f"Smart Sync part {idx} render failed: {(res.stderr or '')[-1500:]}")
        parts.append(dst)
        if progress:
            progress(f"Smart Sync video {idx}/{total}", 5.0 + 70.0 * idx / total)

    list_file = work_dir / "smart_sync_video_parts.txt"
    list_file.write_text("\n".join("file '" + str(p.resolve()).replace("\\", "/").replace("'", "'\\''") + "'"
                                   for p in parts) + "\n", encoding="utf-8")
    silent = work_dir / "smart_sync_video_noaudio.mp4"
    res = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                          "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(silent)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Smart Sync concat failed: {(res.stderr or '')[-1500:]}")

    v_len, a_len = _probe_duration(silent), _probe_duration(tts_audio_path)
    if progress:
        progress(f"SYNC check: picture={v_len:.2f}s narration={a_len:.2f}s diff={v_len - a_len:+.3f}s", 80.0)

    post = []
    if enable_4k_filter:
        post += ["eq=contrast=1.08:brightness=0.02:saturation=1.08", "unsharp=5:5:0.45:5:5:0.0"]
    if mirror_mode_7s:
        post.append("hflip=enable='gte(mod(t,14),7)'")
    input_args = ["-i", str(silent), "-i", str(tts_audio_path)]
    audio_map = "1:a:0"
    filter_complex = None
    if background_audio_path is not None:
        background_audio_path = Path(background_audio_path)
        if not background_audio_path.is_file():
            raise FileNotFoundError(f"Background audio not found: {background_audio_path}")
        if not segments:
            raise ValueError("Smart Sync needs at least one segment to align background audio.")
        input_args += ["-i", str(background_audio_path)]
        audio_filters = []
        bg_labels = []
        for index, seg in enumerate(segments):
            source_start = max(0.0, float(seg["video_start"]))
            source_end = max(source_start + 0.04, float(seg["video_end"]))
            source_duration = source_end - source_start
            target_duration = max(0.04, float(seg["slot_duration"]))
            bg_label = f"bg{index}"
            bg_labels.append(f"[{bg_label}]")
            video_clips = seg.get("video_clips") or []
            if video_clips:
                clip_labels = []
                for clip_index, clip in enumerate(video_clips):
                    clip_label = f"bg{index}clip{clip_index}"
                    clip_labels.append(f"[{clip_label}]")
                    audio_filters.append(
                        f"[2:a]atrim=start={float(clip['start']):.6f}:end={float(clip['end']):.6f},"
                        f"asetpts=PTS-STARTPTS,aresample=44100[{clip_label}]"
                    )
                if len(clip_labels) > 1:
                    audio_filters.append(
                        f"{''.join(clip_labels)}concat=n={len(clip_labels)}:v=0:a=1[bgraw{index}]"
                    )
                    raw_label = f"[bgraw{index}]"
                else:
                    raw_label = clip_labels[0]
                audio_filters.append(
                    f"{raw_label}apad=whole_dur={target_duration:.6f},"
                    f"atrim=duration={target_duration:.6f}[{bg_label}]"
                )
            else:
                audio_filters.append(
                    f"[2:a]atrim=start={source_start:.6f}:end={source_end:.6f},"
                    f"asetpts=PTS-STARTPTS,{_atempo_chain(source_duration / target_duration)},"
                    f"apad=whole_dur={target_duration:.6f},atrim=duration={target_duration:.6f}[{bg_label}]"
                )
        if len(bg_labels) > 1:
            audio_filters.append(
                f"{''.join(bg_labels)}concat=n={len(bg_labels)}:v=0:a=1[bg_joined]"
            )
            joined_label = "[bg_joined]"
        else:
            joined_label = bg_labels[0]
        audio_filters.extend([
            f"{joined_label}apad=whole_dur={v_len:.6f},atrim=duration={v_len:.6f},volume=0.7[bg]",
            "[1:a]asetpts=PTS-STARTPTS[tts]",
            "[bg][tts]amix=inputs=2:duration=longest:dropout_transition=0.2:normalize=0[a]",
        ])
        filter_complex = ";".join(audio_filters)
        audio_map = "[a]"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *input_args]
    if filter_complex:
        cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "0:v:0", "-map", audio_map]
    if post:
        cmd += ["-vf", ",".join(post), *(cpu_args if use_cpu else encoder_args)]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 and post:
        fallback_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *input_args]
        if filter_complex:
            fallback_cmd += ["-filter_complex", filter_complex]
        fallback_cmd += ["-map", "0:v:0", "-map", audio_map, "-vf", ",".join(post),
                         *cpu_args, "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                         str(output_path)]
        res = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Smart Sync final mux failed: {(res.stderr or '')[-1500:]}")
    if progress:
        progress("Smart Sync ပြီးပါပြီ။", 100.0)
    return output_path
