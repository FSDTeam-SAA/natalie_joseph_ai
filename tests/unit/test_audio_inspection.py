from __future__ import annotations

import struct

import pytest

from app.voice.audio_inspection import AudioInspectionError, inspect_audio_duration


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def _ebml(element_id: bytes, payload: bytes) -> bytes:
    assert len(payload) < 127
    return element_id + bytes((0x80 | len(payload),)) + payload


def test_inspects_pcm_wav_duration() -> None:
    sample_rate = 8000
    duration = 2.0
    data_size = int(sample_rate * duration) * 2
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", data_size) + bytes(data_size)
    wav = b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks

    assert inspect_audio_duration(wav, "audio/wav") == pytest.approx(duration)
    assert inspect_audio_duration(wav, "audio/x-wav") == pytest.approx(duration)


def test_inspects_mp3_frames_instead_of_estimating_from_file_size() -> None:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz: 417-byte frames, 1152 samples each.
    frame_count = 80
    frame = b"\xff\xfb\x90\x00" + bytes(417 - 4)
    mp3 = frame * frame_count

    assert inspect_audio_duration(mp3, "audio/mpeg") == pytest.approx(
        frame_count * 1152 / 44100
    )


@pytest.mark.parametrize("mime_type", ["audio/mp4", "audio/m4a"])
def test_inspects_mp4_movie_timeline(mime_type: str) -> None:
    ftyp = _box(b"ftyp", b"M4A \x00\x00\x00\x00M4A ")
    mvhd = _box(b"mvhd", struct.pack(">IIIII", 0, 0, 0, 1000, 3500))
    audio = ftyp + _box(b"moov", mvhd)

    assert inspect_audio_duration(audio, mime_type) == pytest.approx(3.5)


def test_inspects_ogg_opus_granule_position() -> None:
    opus_head = b"OpusHead" + bytes((1, 2)) + struct.pack("<H", 312) + struct.pack("<I", 48000)
    opus_head += struct.pack("<hB", 0, 0)
    granule = 96000 + 312
    page = b"OggS" + bytes((0, 2)) + struct.pack("<QIIIB", granule, 7, 0, 0, 1)
    page += bytes((len(opus_head),)) + opus_head

    assert inspect_audio_duration(page, "audio/ogg") == pytest.approx(2.0)


def test_inspects_webm_duration_from_info() -> None:
    timestamp_scale = _ebml(b"\x2a\xd7\xb1", (1_000_000).to_bytes(3, "big"))
    duration = _ebml(b"\x44\x89", struct.pack(">d", 4250.0))
    info = _ebml(b"\x15\x49\xa9\x66", timestamp_scale + duration)
    segment = _ebml(b"\x18\x53\x80\x67", info)
    ebml_header = _ebml(b"\x1a\x45\xdf\xa3", b"")

    assert inspect_audio_duration(ebml_header + segment, "audio/webm") == pytest.approx(4.25)


@pytest.mark.parametrize(
    ("data", "mime_type"),
    [(b"RIFF\x04\x00\x00\x00WAVE", "audio/wav"), (b"OggS", "audio/ogg")],
)
def test_malformed_or_indeterminate_audio_fails_closed(data: bytes, mime_type: str) -> None:
    with pytest.raises(AudioInspectionError):
        inspect_audio_duration(data, mime_type)
