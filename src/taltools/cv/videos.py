import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from os import path as osp

import numpy as np
import cv2
import ffmpeg

from taltools.io.files import init_directories
from taltools.logging import PrintLogger


MAX_DURATION = 3 * 3600  # ~3h in seconds; longer videos are treated as bugged/irrelevant
# Relative gap between the header's nominal fps and the effective fps
# (frame_count / duration) beyond which we warn. Clinical captures routinely
# label a nominal 25/1 while actually running at ~25.03 fps; that ~0.13% drift is
# enough to misplace late-video annotations by several frames.
_FPS_REL_TOLERANCE = 0.005
logger = PrintLogger(__name__)

# Env-var override names checked first by resolve_binary, e.g. FFPROBE_BINARY / FFMPEG_BINARY.
_BINARY_ENV_VARS = {
    'ffprobe': 'FFPROBE_BINARY',
    'ffmpeg': 'FFMPEG_BINARY',
}


def _sys_prefix_candidates(name):
    """Locations of `name` inside the active conda/venv (sys.prefix), per-OS."""
    if os.name == 'nt':  # Windows
        exe = name + '.exe'
        return [
            osp.join(sys.prefix, 'Scripts', exe),
            osp.join(sys.prefix, 'Library', 'bin', exe),  # conda's ffmpeg/ffprobe live here
            osp.join(sys.prefix, exe),
        ]
    return [osp.join(sys.prefix, 'bin', name)]  # POSIX (Linux/HPC, macOS)


@lru_cache(maxsize=None)
def resolve_binary(name):
    """
    Resolve an absolute path to `name` ('ffprobe' or 'ffmpeg'), robust across
    environments. Priority:
      1. Explicit env-var override (FFPROBE_BINARY / FFMPEG_BINARY, generic
         <NAME>_BINARY otherwise).
      2. The active conda/venv via sys.prefix. This is the key fix for clusters
         where the env's bin dir is not on the subprocess PATH even though the
         binary is installed in the active env.
      3. shutil.which(name) on PATH.
      4. Bare name as a last resort (the downstream call raises a clear error).
    Result is cached per name; call resolve_binary.cache_clear() if env changes.
    """
    env_var = _BINARY_ENV_VARS.get(name, f'{name.upper()}_BINARY')
    override = os.environ.get(env_var)
    if override:
        if osp.isfile(override) and os.access(override, os.X_OK):
            logger.debug(f"resolve_binary({name}): using env override {env_var}={override}")
            return override
        logger.warning(f"resolve_binary({name}): {env_var}={override} is not an executable file; ignoring")

    for candidate in _sys_prefix_candidates(name):
        if osp.isfile(candidate) and os.access(candidate, os.X_OK):
            logger.debug(f"resolve_binary({name}): found in sys.prefix at {candidate}")
            return candidate

    found = shutil.which(name)
    if found:
        logger.debug(f"resolve_binary({name}): found on PATH at {found}")
        return found

    logger.warning(f"resolve_binary({name}): not found via override/sys.prefix/PATH; falling back to bare '{name}'")
    return name


def _parse_rational(value):
    """
    Parse ffprobe rate strings like '30000/1001', '25/1', '25', '0/0', 'N/A'.
    Returns a positive float, or None if not parseable / non-positive.
    """
    if value is None:
        return None
    value = str(value).strip()
    if value in ('', 'N/A', '0/0'):
        return None
    try:
        if '/' in value:
            num, den = value.split('/', 1)
            num, den = float(num), float(den)
            if den == 0:
                return None
            result = num / den
        else:
            result = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def _parse_float(value):
    """
    Parse a numeric metadata field. Returns a non-negative float, or None for
    None / 'N/A' / '' / unparseable / negative sentinels (e.g. nb_frames=-1).
    """
    if value is None:
        return None
    value = str(value).strip()
    if value in ('', 'N/A'):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if result >= 0 else None


def take_subclip(video_path, start_time, end_time, fps, out_path):
    ffmpeg.input(video_path).video \
        .trim(start_frame=start_time, end_frame=end_time) \
        .setpts('PTS-STARTPTS') \
        .filter('fps', fps=fps, round='up') \
        .output(out_path) \
        .run(cmd=resolve_binary('ffmpeg'))


def video2img(video_path, out_path):
    name = osp.splitext(osp.basename(video_path))[0]
    init_directories(out_path)
    cap = cv2.VideoCapture(video_path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    d = len(str(n))
    i, ret = 0, True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(osp.join(out_path, f'{name}_{str(i).zfill(d)}.jpg'), frame)
        i += 1
    cap.release()


def fps2time(frame_num, fps):
    x = frame_num / fps
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes:02d}:{seconds:02d}'


def _ffmpeg_probe(filename):
    """Fast header read via ffmpeg-python. Handles healthy files (common case)."""
    vinf = ffmpeg.probe(filename, cmd=resolve_binary('ffprobe'))
    video_streams = [s for s in vinf.get('streams', []) if s.get('codec_type') == 'video']
    if not video_streams:
        raise ValueError(f"ffmpeg.probe found no video stream for: {filename}")
    s = video_streams[0]
    if 'width' not in s or 'height' not in s:
        raise ValueError(f"ffmpeg.probe found no resolution for: {filename}")
    width, height = int(s['width']), int(s['height'])

    fps = _parse_rational(s.get('avg_frame_rate')) or _parse_rational(s.get('r_frame_rate'))
    # Duration: prefer the video stream's, fall back to the container's (mp4 often stores it there).
    length = _parse_float(s.get('duration')) or _parse_float(vinf.get('format', {}).get('duration'))
    nb = _parse_float(s.get('nb_frames'))
    frame_count = int(nb) if nb else (int(np.ceil(length * fps)) if length and fps else None)

    return width, height, fps, frame_count, length


def _ffprobe_fallback(filename):
    """Fast header read via the ffprobe binary directly (no decoding)."""
    result = subprocess.run([
        resolve_binary('ffprobe'), '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
        '-of', 'json', filename
    ], capture_output=True, text=True, timeout=90)
    info = json.loads(result.stdout or '{}')
    streams = info.get('streams', [])
    if not streams:
        raise ValueError(f"ffprobe found no video streams in: {filename}")
    s = streams[0]
    if 'width' not in s or 'height' not in s:
        raise ValueError(f"ffprobe found no resolution in: {filename}")
    width, height = int(s['width']), int(s['height'])
    fps = _parse_rational(s.get('avg_frame_rate')) or _parse_rational(s.get('r_frame_rate'))
    duration = _parse_float(s.get('duration'))
    nb = _parse_float(s.get('nb_frames'))
    frame_count = int(nb) if nb else (round(duration * fps) if duration and fps else None)
    return width, height, fps, frame_count, duration


def _ffprobe_count(filename):
    """
    Damaged-header recovery: count packets and derive duration from
    frame_count / fps when the header lacks it. Demux-only, so far cheaper than
    a cv2 full decode but slower than a header read.
    """
    result = subprocess.run([
        resolve_binary('ffprobe'), '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,duration',
        '-of', 'json', filename
    ], capture_output=True, text=True, timeout=600)
    info = json.loads(result.stdout or '{}')
    streams = info.get('streams', [])
    if not streams:
        raise ValueError(f"ffprobe(-count_packets) found no video streams in: {filename}")
    s = streams[0]
    if 'width' not in s or 'height' not in s:
        raise ValueError(f"ffprobe(-count_packets) found no resolution in: {filename}")
    width, height = int(s['width']), int(s['height'])
    fps = _parse_rational(s.get('r_frame_rate')) or _parse_rational(s.get('avg_frame_rate'))
    frame_count = _parse_float(s.get('nb_read_packets'))
    frame_count = int(frame_count) if frame_count else None
    duration = _parse_float(s.get('duration'))
    if duration is None and frame_count and fps:
        duration = frame_count / fps
    if frame_count is None and duration and fps:
        frame_count = round(duration * fps)
    return width, height, fps, frame_count, duration


def _cv2_fallback(filename):
    cap = cv2.VideoCapture(filename)
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count / fps > MAX_DURATION:
                raise ValueError(f"Video exceeds maximum duration ({MAX_DURATION}s): {filename}")
        length = frame_count / fps
        return int(width), int(height), fps, frame_count, length
    finally:
        cap.release()


def _is_plausible(width, height, fps, frame_count, length):
    return bool(width and height and fps and frame_count and length
                and width > 0 and height > 0 and fps > 0 and frame_count > 0 and length > 0)


def _header_meta(filename):
    """Cheap header read for the fields we can trust from a header: width,
    height, a *nominal* fps, and duration. Returns
    (width, height, nominal_fps, duration); any field may be None except
    width/height (a probe that yields no resolution raises). Tries the
    ffmpeg-python wrapper first, then the ffprobe binary directly.

    NOTE: the frame_count these probers return is deliberately discarded here —
    when the header lacks nb_frames it is fabricated from duration * nominal_fps,
    which is exactly the value we must not trust. The authoritative count comes
    from packet counting in get_video_properties.
    """
    last_err = None
    for prober in (_ffmpeg_probe, _ffprobe_fallback):
        try:
            width, height, fps, _frame_count, duration = prober(filename)
            return width, height, fps, duration
        except Exception as e:
            logger.debug(f"{prober.__name__} header read failed for {filename}: {e}")
            last_err = e
    raise RuntimeError(f"header read failed for {filename}: {last_err}")


# Ordered schema of the per-video metadata dict returned by get_video_properties.
# Persisted (e.g. as NADI's extended videos table) and reused for column ordering.
CANONICAL_COLUMNS = [
    'basename', 'video_path', 'width', 'height',
    'avg_frame_rate', 'r_frame_rate', 'nominal_fps',
    'nb_frames_header', 'nb_read_packets', 'cv2_decoded',
    'duration', 'effective_fps', 'fps_delta',
    'header_count_mismatch', 'fps_mismatch', 'flagged', 'error',
    'n_pts', 'first_pts', 'last_pts', 'pts_span', 'fps_pts',
    'duration_pts', 'duration_vs_pts_delta', 'fps_pts_confirms_effective',
]


def _probe(filename):
    """One ffprobe demux: header fields, packet count, AND per-packet PTS.

    A single ``-count_packets`` pass that also emits ``packet=pts_time``, so the
    frame count (``nb_read_packets``) and the PTS-derived fps come from the SAME
    demux — no second pass. Returns a dict of raw header fields plus
    ``nb_read_packets`` and ``pts`` (list of parseable timestamps in seconds).
    Raises if there is no video stream or no resolution.
    """
    result = subprocess.run([
        resolve_binary('ffprobe'), '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration'
        ':format=duration:packet=pts_time',
        '-of', 'json', filename,
    ], capture_output=True, text=True, timeout=MAX_DURATION)
    info = json.loads(result.stdout or '{}')
    streams = info.get('streams', [])
    if not streams:
        raise ValueError(f"ffprobe found no video streams in: {filename}")
    s = streams[0]
    fmt = info.get('format', {})
    if 'width' not in s or 'height' not in s:
        raise ValueError(f"ffprobe found no resolution in: {filename}")

    nb_frames = _parse_float(s.get('nb_frames'))
    packets = _parse_float(s.get('nb_read_packets'))
    pts = [v for v in (_parse_float(p.get('pts_time'))
                       for p in info.get('packets', [])) if v is not None]
    return {
        'width': int(s['width']),
        'height': int(s['height']),
        'avg_frame_rate': _parse_rational(s.get('avg_frame_rate')),
        'r_frame_rate': _parse_rational(s.get('r_frame_rate')),
        'nb_frames_header': int(nb_frames) if nb_frames else None,
        'nb_read_packets': int(packets) if packets else None,
        'duration': _parse_float(s.get('duration')) or _parse_float(fmt.get('duration')),
        'pts': pts,
    }


def _pts_stats(pts):
    """fps derived directly from frame spacing — independent of the header
    `duration` field and the nominal rate. fps_pts is None with < 2 timestamps.
    """
    if len(pts) < 2:
        return {'n_pts': len(pts), 'first_pts': None, 'last_pts': None,
                'pts_span': None, 'fps_pts': None}
    first_pts, last_pts = min(pts), max(pts)
    span = last_pts - first_pts
    # N timestamps span (N-1) inter-frame intervals.
    fps_pts = (len(pts) - 1) / span if span > 0 else None
    return {'n_pts': len(pts), 'first_pts': first_pts, 'last_pts': last_pts,
            'pts_span': span, 'fps_pts': fps_pts}


def _empty_row(filename):
    """A CANONICAL_COLUMNS dict with everything NaN, identity fields filled."""
    row = {c: np.nan for c in CANONICAL_COLUMNS}
    row['basename'] = osp.splitext(osp.basename(filename))[0]
    row['video_path'] = filename
    return row


def get_video_properties(filename, *, validate_decode=False):
    """Probe a video and return the full per-video metadata dict.

    Keys are :data:`CANONICAL_COLUMNS`. Missing values are ``np.nan``; the boolean
    flags are Python bools; ``error`` is ``np.nan`` when none. The function never
    raises — any failure is captured into ``error`` and a partial row returned, so
    one call yields exactly one row for a whole-database metadata table.

    Fields are raw measurements (no fabrication):
      - frame count is the demuxed packet count (``nb_read_packets``);
      - ``effective_fps`` is strictly ``nb_read_packets / duration``;
      - the PTS cross-check (``fps_pts`` and the ``duration_*`` / confirms columns)
        always runs, deriving fps straight from frame timestamps;
      - ``cv2_decoded`` is filled only when ``validate_decode`` is True or the
        packet count is unavailable (full-decode recovery) — decoding is otherwise
        skipped as too expensive.

    Callers that need the legacy ``(width, height, fps, frame_count, length)``
    tuple should compose it from this dict (frame_count = ``nb_read_packets`` else
    ``cv2_decoded``; fps = frame_count / duration).
    """
    row = _empty_row(filename)
    try:
        if not osp.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # 1. One demux pass: header fields, packet count, and per-packet PTS.
        probe = _probe(filename)
        pts = probe.pop('pts', [])
        row.update(probe)

        nominal_fps = row['avg_frame_rate'] or row['r_frame_rate']
        row['nominal_fps'] = nominal_fps if nominal_fps else np.nan
        packets, duration = row['nb_read_packets'], row['duration']
        effective_fps = (packets / duration) if (packets and duration and duration > 0) else None
        row['effective_fps'] = effective_fps if effective_fps is not None else np.nan
        row['fps_delta'] = (effective_fps - nominal_fps) \
            if (effective_fps is not None and nominal_fps) else np.nan

        row['header_count_mismatch'] = bool(
            row['nb_frames_header'] is not None
            and not (isinstance(row['nb_frames_header'], float) and np.isnan(row['nb_frames_header']))
            and packets is not None and int(row['nb_frames_header']) != packets)
        row['fps_mismatch'] = bool(
            effective_fps is not None and nominal_fps
            and abs(effective_fps - nominal_fps) / nominal_fps > _FPS_REL_TOLERANCE)
        row['flagged'] = bool(
            row['nb_frames_header'] is None
            or (isinstance(row['nb_frames_header'], float) and np.isnan(row['nb_frames_header']))
            or row['header_count_mismatch'] or row['fps_mismatch'])

        # 2. MAX_DURATION short-circuit: record, flag via error, skip PTS/cv2.
        if duration is not None and duration > MAX_DURATION:
            row['error'] = f"Video exceeds maximum duration ({MAX_DURATION}s): length={duration}"
            return row

        # 3. PTS cross-check (from the same demux — no extra pass).
        row.update(_pts_stats(pts))
        fps_pts, n_pts, span = row['fps_pts'], row['n_pts'], row['pts_span']
        if fps_pts is not None and n_pts and n_pts > 1 and span:
            duration_pts = span * n_pts / (n_pts - 1)
            row['duration_pts'] = duration_pts
            if duration is not None:
                row['duration_vs_pts_delta'] = duration - duration_pts
            if effective_fps is not None and nominal_fps:
                row['fps_pts_confirms_effective'] = bool(
                    abs(fps_pts - effective_fps) < abs(fps_pts - nominal_fps))

        # 4. cv2 decode: only on request or as packet-count recovery.
        if validate_decode or packets is None:
            try:
                _w, _h, _fps, decoded, _len = _cv2_fallback(filename)
                row['cv2_decoded'] = decoded
                if not row['width']:
                    row['width'] = _w
                if not row['height']:
                    row['height'] = _h
            except Exception as e:
                row['error'] = str(e)

    except Exception as e:
        row['error'] = str(e)
        logger.warning(f"get_video_properties failed for {filename}: {e}")

    # Normalise None -> NaN so the row is uniform for tabular storage.
    for k, v in row.items():
        if v is None:
            row[k] = np.nan
    return row
