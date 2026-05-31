from time import perf_counter

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import fftconvolve


class AudioScan:
    def __init__(
        self,
        duration_sec,
        acoustic_events_detected,
        intervals,
        analysis_runtime_sec,
        output_wav,
    ):
        self.duration_sec = duration_sec
        self.acoustic_events_detected = acoustic_events_detected
        self.intervals = intervals
        self.analysis_runtime_sec = analysis_runtime_sec
        self.output_wav = output_wav


def _extract_audio(video_path, output_wav):
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for audio-guided processing. "
            "Run: pip install -r requirements.txt"
        ) from exc

    video = VideoFileClip(str(video_path))

    try:
        if video.audio is None:
            raise RuntimeError("Video file does not contain an audio track.")

        video.audio.write_audiofile(
            str(output_wav),
            fps=22050,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )
    finally:
        video.close()


def _find_loud_segments(wav_path, threshold_db, min_event_seconds):
    sample_rate, data = wavfile.read(str(wav_path))

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data.astype(float, copy=False)

    if len(data) == 0:
        return 0.0, []

    duration_sec = len(data) / sample_rate
    max_value = np.max(np.abs(data))
    normalized = data / max_value if max_value > 0 else data
    window_size = max(1, int(sample_rate * 0.1))
    window = np.ones(window_size) / window_size
    mean_square = fftconvolve(normalized**2, window, mode="same")
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    is_loud = 20 * np.log10(np.clip(rms, 1e-3, 1.0)) > threshold_db
    changes = np.diff(is_loud.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if is_loud[0]:
        starts = np.insert(starts, 0, 0)

    if is_loud[-1]:
        ends = np.append(ends, len(is_loud) - 1)

    segments = []

    for start, end in zip(starts, ends):
        start_sec = float(start / sample_rate)
        end_sec = float(end / sample_rate)

        if end_sec - start_sec >= min_event_seconds:
            segments.append((start_sec, end_sec))

    return duration_sec, segments


def _merge_and_pad_segments(segments, duration_sec, padding_seconds):
    padded = sorted(
        (
            max(0.0, start - padding_seconds),
            min(duration_sec, end + padding_seconds),
        )
        for start, end in segments
    )
    merged = []

    for start, end in padded:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue

        merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def analyze_audio(
    video_path,
    output_dir,
    threshold_db,
    min_event_seconds,
    padding_seconds,
):
    start_time = perf_counter()
    output_wav = output_dir / f"{video_path.stem.replace(' ', '_')}_audio.wav"
    _extract_audio(video_path, output_wav)
    duration_sec, segments = _find_loud_segments(
        wav_path=output_wav,
        threshold_db=threshold_db,
        min_event_seconds=min_event_seconds,
    )
    intervals = _merge_and_pad_segments(
        segments=segments,
        duration_sec=duration_sec,
        padding_seconds=padding_seconds,
    )

    return AudioScan(
        duration_sec=duration_sec,
        acoustic_events_detected=len(segments),
        intervals=intervals,
        analysis_runtime_sec=perf_counter() - start_time,
        output_wav=output_wav,
    )
