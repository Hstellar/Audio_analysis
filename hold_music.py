import webrtcvad
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ---------------- CONFIG ---------------- #

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2

MIN_NONSILENT_MS = 3000
SILENCE_THRESH_DB = -40
MIN_SILENCE_LEN_MS = 500

HOLD_SPEECH_RATIO_THRESHOLD = 0.15
MERGE_GAP_MS = 500

EXPORT_AUDIO = True

# ---------------------------------------- #


def load_audio(path):
    audio = (
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .set_sample_width(2)  # 16-bit PCM
    )
    return audio


def compute_speech_flags(audio):
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    frame_bytes = int(SAMPLE_RATE * 2 * FRAME_MS / 1000)
    raw = audio.raw_data

    flags = []
    for i in range(0, len(raw) - frame_bytes, frame_bytes):
        frame = raw[i:i + frame_bytes]
        flags.append(vad.is_speech(frame, SAMPLE_RATE))

    return flags


def speech_ratio(start_ms, end_ms, speech_flags):
    start_f = int(start_ms / FRAME_MS)
    end_f = int(end_ms / FRAME_MS)

    frames = speech_flags[start_f:end_f]
    if not frames:
        return 0.0

    return sum(frames) / len(frames)


def detect_hold_music(audio, speech_flags):
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB
    )

    hold_candidates = []

    for start_ms, end_ms in nonsilent_ranges:
        duration = end_ms - start_ms
        if duration < MIN_NONSILENT_MS:
            continue

        ratio = speech_ratio(start_ms, end_ms, speech_flags)

        if ratio <= HOLD_SPEECH_RATIO_THRESHOLD:
            hold_candidates.append((start_ms, end_ms))

    # Merge nearby segments
    merged = []
    for start, end in hold_candidates:
        if not merged:
            merged.append([start, end])
        elif start - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(x) for x in merged]


def main(audio_path):
    audio = load_audio(audio_path)
    speech_flags = compute_speech_flags(audio)
    hold_segments = detect_hold_music(audio, speech_flags)

    if not hold_segments:
        print("No hold music detected.")
        return

    hold_start, hold_end = max(
        hold_segments,
        key=lambda x: x[1] - x[0]
    )

    duration_sec = (hold_end - hold_start) / 1000

    print(f"Hold music start: {hold_start} ms")
    print(f"Hold music end:   {hold_end} ms")
    print(f"Max duration:     {duration_sec:.2f} sec")

    if EXPORT_AUDIO:
        hold_audio = audio[hold_start:hold_end]
        hold_audio.export("hold_music.wav", format="wav")
        print("Hold music audio exported: hold_music.wav")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hold_music_detector.py <audio_file>")
        sys.exit(1)

    main(sys.argv[1])
