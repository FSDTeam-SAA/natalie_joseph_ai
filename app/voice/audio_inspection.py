"""Bounded, dependency-free duration inspection for accepted voice uploads.

The service must reject overlong audio *before* sending it to paid STT.  File
size is not a duration proxy (low-bitrate audio can be very long), so each
accepted container is inspected using its native timing metadata/frames.
Malformed or indeterminate files fail closed.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator


class AudioInspectionError(ValueError):
    """The upload is malformed or its duration cannot be determined safely."""


def inspect_audio_duration(data: bytes, mime_type: str) -> float:
    """Return duration in seconds for every MIME type accepted by VoiceService."""
    canonical = "audio/wav" if mime_type == "audio/x-wav" else mime_type
    inspectors = {
        "audio/wav": _wav_duration,
        "audio/mpeg": _mp3_duration,
        "audio/mp4": _mp4_duration,
        "audio/m4a": _mp4_duration,
        "audio/ogg": _ogg_duration,
        "audio/webm": _webm_duration,
    }
    inspector = inspectors.get(canonical)
    if inspector is None:
        raise AudioInspectionError("unsupported audio container")
    try:
        duration = inspector(data)
    except (IndexError, OverflowError, struct.error) as exc:
        raise AudioInspectionError("malformed audio container") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioInspectionError("audio duration is missing or invalid")
    return duration


def _wav_duration(data: bytes) -> float:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioInspectionError("invalid WAV header")

    sample_rate = 0
    byte_rate = 0
    fact_samples: int | None = None
    data_bytes: int | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(data):
            raise AudioInspectionError("truncated WAV chunk")
        if chunk_id == b"fmt " and chunk_size >= 16:
            sample_rate = int.from_bytes(data[payload_start + 4 : payload_start + 8], "little")
            byte_rate = int.from_bytes(data[payload_start + 8 : payload_start + 12], "little")
        elif chunk_id == b"fact" and chunk_size >= 4:
            fact_samples = int.from_bytes(data[payload_start : payload_start + 4], "little")
        elif chunk_id == b"data":
            data_bytes = chunk_size
        offset = payload_end + (chunk_size & 1)

    if fact_samples is not None and sample_rate > 0:
        return fact_samples / sample_rate
    if data_bytes is not None and byte_rate > 0:
        return data_bytes / byte_rate
    raise AudioInspectionError("WAV timing chunks are missing")


_MPEG1_BITRATES = {
    1: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
}
_MPEG2_BITRATES = {
    1: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    3: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}


def _skip_id3v2(data: bytes) -> int:
    if not data.startswith(b"ID3"):
        return 0
    if len(data) < 10 or any(byte & 0x80 for byte in data[6:10]):
        raise AudioInspectionError("invalid ID3 tag")
    size = sum(byte << shift for byte, shift in zip(data[6:10], (21, 14, 7, 0), strict=True))
    footer_size = 10 if data[5] & 0x10 else 0
    offset = 10 + size + footer_size
    if offset > len(data):
        raise AudioInspectionError("truncated ID3 tag")
    return offset


def _mp3_duration(data: bytes) -> float:
    offset = _skip_id3v2(data)
    duration = 0.0
    frames = 0
    while offset + 4 <= len(data):
        header = int.from_bytes(data[offset : offset + 4], "big")
        if header & 0xFFE00000 != 0xFFE00000:
            # ID3v1 and APE tags are legal after the MPEG audio frames.
            has_trailing_tag = data[offset : offset + 3] == b"TAG" or data[offset:].startswith(
                b"APETAGEX"
            )
            if frames and has_trailing_tag:
                break
            raise AudioInspectionError("invalid MPEG frame sync")
        version_bits = (header >> 19) & 0b11
        layer_bits = (header >> 17) & 0b11
        bitrate_index = (header >> 12) & 0xF
        sample_rate_index = (header >> 10) & 0b11
        padding = (header >> 9) & 1
        if version_bits == 0b01 or layer_bits == 0 or bitrate_index in {0, 15}:
            raise AudioInspectionError("unsupported MPEG frame header")
        if sample_rate_index == 0b11:
            raise AudioInspectionError("invalid MPEG sample rate")

        layer = 4 - layer_bits
        base_rate = (44100, 48000, 32000)[sample_rate_index]
        divisor = 1 if version_bits == 0b11 else 2 if version_bits == 0b10 else 4
        sample_rate = base_rate // divisor
        bitrate_table = _MPEG1_BITRATES if version_bits == 0b11 else _MPEG2_BITRATES
        bitrate = bitrate_table[layer][bitrate_index] * 1000

        if layer == 1:
            frame_size = ((12 * bitrate // sample_rate) + padding) * 4
            samples = 384
        else:
            coefficient = 72 if layer == 3 and version_bits != 0b11 else 144
            frame_size = coefficient * bitrate // sample_rate + padding
            samples = 1152 if layer == 2 or version_bits == 0b11 else 576
        if frame_size < 4 or offset + frame_size > len(data):
            raise AudioInspectionError("truncated MPEG audio frame")
        duration += samples / sample_rate
        frames += 1
        offset += frame_size

    if not frames:
        raise AudioInspectionError("no MPEG audio frames found")
    # Zero padding after the last frame is common; other unparsed bytes are not.
    if offset < len(data) and any(data[offset:]):
        raise AudioInspectionError("unexpected data after MPEG audio frames")
    return duration


def _iter_mp4_boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    offset = start
    while offset < end:
        if offset + 8 > end:
            raise AudioInspectionError("truncated MP4 box header")
        box_size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header_size = 8
        if box_size == 1:
            if offset + 16 > end:
                raise AudioInspectionError("truncated extended MP4 box header")
            box_size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        elif box_size == 0:
            box_size = end - offset
        if box_size < header_size or offset + box_size > end:
            raise AudioInspectionError("invalid MP4 box size")
        yield box_type, offset + header_size, offset + box_size
        offset += box_size


def _mp4_fullbox_duration(data: bytes, start: int, end: int, box_type: bytes) -> float:
    if end - start < 20:
        raise AudioInspectionError("truncated MP4 timing box")
    version = data[start]
    if version == 0:
        timescale_offset = start + 12
        duration_offset = start + 16
        duration_size = 4
    elif version == 1:
        timescale_offset = start + 20
        duration_offset = start + 24
        duration_size = 8
    else:
        raise AudioInspectionError("unsupported MP4 timing version")
    if duration_offset + duration_size > end:
        raise AudioInspectionError("truncated MP4 timing values")
    timescale = int.from_bytes(data[timescale_offset : timescale_offset + 4], "big")
    duration = int.from_bytes(data[duration_offset : duration_offset + duration_size], "big")
    unknown_duration = (1 << (duration_size * 8)) - 1
    if timescale == 0 or duration in {0, unknown_duration}:
        raise AudioInspectionError(f"invalid {box_type.decode('ascii')} duration")
    return duration / timescale


def _mp4_duration(data: bytes) -> float:
    durations: list[float] = []
    moov: tuple[int, int] | None = None
    for box_type, start, end in _iter_mp4_boxes(data, 0, len(data)):
        if box_type == b"moov":
            moov = (start, end)
            break
    if moov is None:
        raise AudioInspectionError("MP4 movie metadata is missing")

    for box_type, start, end in _iter_mp4_boxes(data, *moov):
        if box_type == b"mvhd":
            durations.append(_mp4_fullbox_duration(data, start, end, box_type))
        elif box_type == b"trak":
            for child_type, child_start, child_end in _iter_mp4_boxes(data, start, end):
                if child_type != b"mdia":
                    continue
                for media_type, media_start, media_end in _iter_mp4_boxes(
                    data, child_start, child_end
                ):
                    if media_type == b"mdhd":
                        durations.append(
                            _mp4_fullbox_duration(data, media_start, media_end, media_type)
                        )
    if not durations:
        raise AudioInspectionError("MP4 timing metadata is missing")
    # Track duration can exceed the movie header (for example with edits), so
    # enforce against the longest declared timeline.
    return max(durations)


def _ogg_codec_timing(packet: bytes) -> tuple[int, int] | None:
    if packet.startswith(b"OpusHead") and len(packet) >= 19:
        return 48000, int.from_bytes(packet[10:12], "little")
    if packet.startswith(b"\x01vorbis") and len(packet) >= 16:
        return int.from_bytes(packet[12:16], "little"), 0
    if packet.startswith(b"Speex   ") and len(packet) >= 40:
        return int.from_bytes(packet[36:40], "little"), 0
    return None


def _ogg_duration(data: bytes) -> float:
    streams: dict[int, dict[str, object]] = {}
    offset = 0
    pages = 0
    while offset < len(data):
        if offset + 27 > len(data) or data[offset : offset + 4] != b"OggS":
            raise AudioInspectionError("invalid Ogg page")
        if data[offset + 4] != 0:
            raise AudioInspectionError("unsupported Ogg bitstream version")
        granule = int.from_bytes(data[offset + 6 : offset + 14], "little")
        serial = int.from_bytes(data[offset + 14 : offset + 18], "little")
        segment_count = data[offset + 26]
        table_start = offset + 27
        table_end = table_start + segment_count
        if table_end > len(data):
            raise AudioInspectionError("truncated Ogg lacing table")
        lacing = data[table_start:table_end]
        payload_end = table_end + sum(lacing)
        if payload_end > len(data):
            raise AudioInspectionError("truncated Ogg page payload")

        state = streams.setdefault(serial, {"packet": bytearray(), "timing": None, "granule": 0})
        cursor = table_end
        for segment_size in lacing:
            packet = state["packet"]
            assert isinstance(packet, bytearray)
            packet.extend(data[cursor : cursor + segment_size])
            cursor += segment_size
            if segment_size < 255:
                if state["timing"] is None:
                    state["timing"] = _ogg_codec_timing(bytes(packet))
                packet.clear()
        if granule != 0xFFFFFFFFFFFFFFFF:
            state["granule"] = max(int(state["granule"]), granule)
        offset = payload_end
        pages += 1

    if not pages:
        raise AudioInspectionError("no Ogg pages found")
    durations: list[float] = []
    for state in streams.values():
        timing = state["timing"]
        if timing is None:
            continue
        sample_rate, pre_skip = timing
        granule = int(state["granule"])
        if sample_rate > 0 and granule > pre_skip:
            durations.append((granule - pre_skip) / sample_rate)
    if not durations:
        raise AudioInspectionError("Ogg audio timing metadata is missing")
    return max(durations)


def _read_ebml_vint(data: bytes, offset: int, *, keep_marker: bool) -> tuple[int, int, bool]:
    if offset >= len(data) or data[offset] == 0:
        raise AudioInspectionError("invalid EBML variable integer")
    marker = 0x80
    length = 1
    while not data[offset] & marker:
        marker >>= 1
        length += 1
    if length > 8 or offset + length > len(data):
        raise AudioInspectionError("truncated EBML variable integer")
    raw = data[offset : offset + length]
    if keep_marker:
        return int.from_bytes(raw, "big"), length, False
    value = raw[0] & (marker - 1)
    for byte in raw[1:]:
        value = (value << 8) | byte
    unknown = value == (1 << (7 * length)) - 1
    return value, length, unknown


def _iter_ebml_elements(
    data: bytes, start: int, end: int
) -> Iterator[tuple[int, int, int, bool]]:
    offset = start
    while offset < end:
        element_id, id_size, _ = _read_ebml_vint(data, offset, keep_marker=True)
        size, size_length, unknown = _read_ebml_vint(
            data, offset + id_size, keep_marker=False
        )
        payload_start = offset + id_size + size_length
        payload_end = end if unknown else payload_start + size
        if payload_end > end:
            raise AudioInspectionError("truncated EBML element")
        yield element_id, payload_start, payload_end, unknown
        if unknown:
            break
        offset = payload_end


def _ebml_uint(data: bytes, start: int, end: int) -> int:
    if end <= start or end - start > 8:
        raise AudioInspectionError("invalid EBML unsigned integer")
    return int.from_bytes(data[start:end], "big")


def _webm_info(data: bytes, start: int, end: int) -> tuple[int, float | None]:
    scale = 1_000_000
    duration: float | None = None
    for element_id, payload_start, payload_end, _ in _iter_ebml_elements(data, start, end):
        if element_id == 0x2AD7B1:  # TimestampScale
            scale = _ebml_uint(data, payload_start, payload_end)
        elif element_id == 0x4489:  # Duration (IEEE-754)
            size = payload_end - payload_start
            if size == 4:
                duration = struct.unpack(">f", data[payload_start:payload_end])[0]
            elif size == 8:
                duration = struct.unpack(">d", data[payload_start:payload_end])[0]
            else:
                raise AudioInspectionError("invalid WebM Duration value")
    if scale <= 0:
        raise AudioInspectionError("invalid WebM TimestampScale")
    return scale, duration


def _webm_block_timecode(data: bytes, start: int, end: int) -> int:
    _, track_size, _ = _read_ebml_vint(data, start, keep_marker=False)
    timecode_start = start + track_size
    if timecode_start + 3 > end:
        raise AudioInspectionError("truncated WebM block")
    return int.from_bytes(data[timecode_start : timecode_start + 2], "big", signed=True)


def _webm_cluster_end_timecode(data: bytes, start: int, end: int) -> int:
    cluster_timecode = 0
    relative_ends: list[int] = []
    for element_id, payload_start, payload_end, _ in _iter_ebml_elements(data, start, end):
        if element_id == 0xE7:  # Timestamp
            cluster_timecode = _ebml_uint(data, payload_start, payload_end)
        elif element_id == 0xA3:  # SimpleBlock
            relative_ends.append(_webm_block_timecode(data, payload_start, payload_end))
        elif element_id == 0xA0:  # BlockGroup
            relative: int | None = None
            block_duration = 0
            for child_id, child_start, child_end, _ in _iter_ebml_elements(
                data, payload_start, payload_end
            ):
                if child_id == 0xA1:
                    relative = _webm_block_timecode(data, child_start, child_end)
                elif child_id == 0x9B:
                    block_duration = _ebml_uint(data, child_start, child_end)
            if relative is not None:
                relative_ends.append(relative + block_duration)
    return cluster_timecode + max(relative_ends, default=0)


def _webm_duration(data: bytes) -> float:
    segment: tuple[int, int] | None = None
    for element_id, start, end, _ in _iter_ebml_elements(data, 0, len(data)):
        if element_id == 0x18538067:  # Segment
            segment = (start, end)
            break
    if segment is None:
        raise AudioInspectionError("WebM Segment is missing")

    scale = 1_000_000
    declared_duration: float | None = None
    last_timecode = 0
    for element_id, start, end, _ in _iter_ebml_elements(data, *segment):
        if element_id == 0x1549A966:  # Info
            scale, declared_duration = _webm_info(data, start, end)
        elif element_id == 0x1F43B675:  # Cluster
            last_timecode = max(last_timecode, _webm_cluster_end_timecode(data, start, end))

    durations: list[float] = []
    if declared_duration is not None and declared_duration > 0:
        durations.append(declared_duration * scale / 1_000_000_000)
    if last_timecode > 0:
        # Clusters do not expose the final codec frame's decoded length. Add a
        # conservative second so MediaRecorder WebM files without Duration are
        # still accepted without underestimating the configured upper bound.
        durations.append(last_timecode * scale / 1_000_000_000 + 1.0)
    if not durations:
        raise AudioInspectionError("WebM timing metadata is missing")
    return max(durations)
