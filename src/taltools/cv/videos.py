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

MAX_DURATION = 2.5 * 3600  # ~2.5h in seconds; longer videos are treated as bugged/irrelevant
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


# Relative gap between the header's nominal fps and the effective fps
# (frame_count / duration) beyond which we warn. Clinical captures routinely
# label a nominal 25/1 while actually running at ~25.03 fps; that ~0.13% drift is
# enough to misplace late-video annotations by several frames.
_FPS_REL_TOLERANCE = 0.0005


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


def get_video_properties(filename):
    """Probe a video, composing each field from its most trustworthy source.

    - width/height/duration/nominal_fps: cheap header read.
    - frame_count: packet count (nb_read_packets) — the only header-derived
      count we trust; full cv2 decode as a last resort.
    - fps: effective rate = frame_count / duration, falling back to the header's
      nominal fps only when duration is unavailable.

    This intentionally always demuxes (one -count_packets pass) rather than
    returning the first "plausible" header result, because a header missing
    nb_frames yields a plausible-but-wrong frame_count (duration * nominal_fps)
    that would otherwise win and skip the accurate path.
    """
    if not osp.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    width = height = nominal_fps = duration = None

    # 1. Cheap header read — trust only width/height/duration/nominal_fps.
    try:
        width, height, nominal_fps, duration = _header_meta(filename)
    except Exception as e:
        logger.warning(f"_header_meta failed for {filename}: {e}")

    # 2. Authoritative frame count via packet demux (also backfills any header
    #    field that was missing above).
    frame_count = None
    try:
        cw, ch, cfps, frame_count, cdur = _ffprobe_count(filename)
        width = width or cw
        height = height or ch
        nominal_fps = nominal_fps or cfps
        duration = duration or cdur
    except Exception as e:
        logger.warning(f"_ffprobe_count failed for {filename}: {e}")

    # 3. Last resort: full decode (recovers count and resolution when both
    #    header and packet probes failed).
    if frame_count is None or not width or not height:
        try:
            dw, dh, dfps, frame_count, ddur = _cv2_fallback(filename)
            width = width or dw
            height = height or dh
            nominal_fps = nominal_fps or dfps
            # cv2's duration is frame_count / nominal_fps (i.e. wrong when the
            # nominal fps is wrong) — only adopt it if we have nothing better.
            duration = duration or ddur
        except Exception as e:
            logger.warning(f"_cv2_fallback failed for {filename}: {e}")

    if frame_count is None:
        raise RuntimeError(f"Unable to determine frame count for {filename}")

    # 4. Effective fps = frame_count / duration; fall back to nominal fps and, if
    #    needed, reconstruct duration from it.
    if duration and duration > 0:
        fps = frame_count / duration
    else:
        fps = nominal_fps
        if fps and fps > 0:
            duration = frame_count / fps

    # 5. Cross-check and warn (do not fail): a large gap flags a broken header.
    if fps and nominal_fps and nominal_fps > 0 \
            and abs(fps - nominal_fps) / nominal_fps > _FPS_REL_TOLERANCE:
        logger.warning(
            f"{filename}: effective fps {fps:.6f} differs from header nominal "
            f"{nominal_fps:.6f} (frame_count={frame_count}, duration={duration}); "
            f"using effective fps.")

    if duration is not None and duration > MAX_DURATION:
        raise ValueError(f"Video exceeds maximum duration ({MAX_DURATION}s): length={duration}")
    if not _is_plausible(width, height, fps, frame_count, duration):
        raise ValueError(
            f"implausible result for {filename} (w={width}, h={height}, fps={fps}, "
            f"frames={frame_count}, len={duration})")

    return int(width), int(height), float(fps), int(frame_count), float(duration)
