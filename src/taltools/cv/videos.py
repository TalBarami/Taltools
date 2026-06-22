import json
import os
import subprocess
import tempfile
from os import path as osp

import numpy as np
import cv2
import ffmpeg

from taltools.io.files import init_directories
from taltools.logging import PrintLogger

MAX_DURATION = 2.5 * 3600  # ~2.5h in seconds; longer videos are treated as bugged/irrelevant
logger = PrintLogger(__name__)


def take_subclip(video_path, start_time, end_time, fps, out_path):
    ffmpeg.input(video_path).video \
        .trim(start_frame=start_time, end_frame=end_time) \
        .setpts('PTS-STARTPTS') \
        .filter('fps', fps=fps, round='up') \
        .output(out_path) \
        .run()


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


def _ffprobe_fallback(filename):
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json', filename
    ], capture_output=True, text=True, timeout=90)
    info = json.loads(result.stdout)
    streams = info.get('streams', [])
    if not streams:
        raise ValueError(f"ffprobe found no video streams in: {filename}")
    s = streams[0]
    width = int(s['width'])
    height = int(s['height'])
    fps = eval(s['r_frame_rate'])
    duration = float(s['duration'])
    frame_count = round(duration * fps)
    return width, height, fps, frame_count, duration


def _ffmpeg_probe(filename):
    vinf = ffmpeg.probe(filename)
    streams = vinf['streams']

    resolution_candidates = [(s['width'], s['height']) for s in streams if 'width' in s and 'height' in s]
    fps_candidates = [s['avg_frame_rate'] for s in streams if 'avg_frame_rate' in s] + \
                     [s['r_frame_rate'] for s in streams if 'r_frame_rate' in s]
    fps_candidates = [x for x in fps_candidates if x != '0/0']

    resolution = resolution_candidates[0] if len(resolution_candidates) > 0 else None
    fps = eval(fps_candidates[0]) if len(fps_candidates) > 0 else None

    length_candidates = [s['duration'] for s in streams if 'duration' in s]
    if 'format' in vinf and 'duration' in vinf['format']:
        length_candidates.append(vinf['format']['duration'])
    length = eval(length_candidates[0]) if len(length_candidates) > 0 else None

    estimated_frame = length * fps if (length is not None and fps is not None) else None
    frame_candidates = [eval(s['nb_frames']) for s in streams if 'nb_frames' in s]
    if estimated_frame is not None:
        frame_candidates = [f for f in frame_candidates if np.abs(f - estimated_frame) < np.min((50, estimated_frame * 0.1))]
    else:
        frame_candidates = []
    frame_count = int(np.max(frame_candidates)) if len(frame_candidates) > 0 else int(np.ceil(length * fps)) if length and fps else None

    if resolution is None:
        raise ValueError(f"ffmpeg.probe found no resolution for: {filename}")
    width, height = resolution
    return width, height, fps, frame_count, length


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
        return width, height, fps, frame_count, length
    finally:
        cap.release()


def get_video_properties(filename):
    if not osp.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    errors = []
    for strategy in (_ffmpeg_probe, _ffprobe_fallback, _cv2_fallback):
        try:
            width, height, fps, frame_count, length = strategy(filename)
            if length is not None and length > MAX_DURATION:
                raise ValueError(f"Video exceeds maximum duration ({MAX_DURATION}s): length={length}")
            return int(width), int(height), fps, frame_count, length
        except Exception as e:
            logger.warning(f"{strategy.__name__} failed for {filename}: {e}")
            errors.append((strategy.__name__, e))
    raise RuntimeError(f"Unable to get video properties for {filename}. Attempts: {errors}")
