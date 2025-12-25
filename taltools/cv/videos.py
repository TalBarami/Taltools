import os
import subprocess
import tempfile
from os import path as osp

import numpy as np
import cv2
import ffmpeg

from taltools.io.files import init_directories


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


def get_video_properties(filename):
    mx_len = 25e4
    if not osp.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    try:
        vinf = ffmpeg.probe(filename)

        resolution_candidates = [(vinf['streams'][i]['width'], vinf['streams'][i]['height']) for i in range(len(vinf['streams'])) if 'width' in vinf['streams'][i].keys() and 'height' in vinf['streams'][i].keys()]
        fps_candidates = [vinf['streams'][i]['avg_frame_rate'] for i in range(len(vinf['streams'])) if 'avg_frame_rate' in vinf['streams'][i].keys()] + \
                         [vinf['streams'][i]['r_frame_rate'] for i in range(len(vinf['streams'])) if 'r_frame_rate' in vinf['streams'][i].keys()]
        fps_candidates = [x for x in fps_candidates if x != '0/0']

        resolution = resolution_candidates[0] if len(resolution_candidates) > 0 else None
        fps = eval(fps_candidates[0]) if len(fps_candidates) > 0 else None
        length_candidates = [vinf['streams'][i]['duration'] for i in range(len(vinf['streams'])) if 'duration' in vinf['streams'][i].keys()]
        if 'format' in vinf.keys() and 'duration' in vinf['format'].keys():
            length_candidates.append(vinf['format']['duration'])
        length = eval(length_candidates[0]) if len(length_candidates) > 0 else None
        if length is not None and fps is not None:
            estimated_frame = length * fps
        frame_candidates = [eval(vinf['streams'][i]['nb_frames']) for i in range(len(vinf['streams'])) if 'nb_frames' in vinf['streams'][i].keys()]
        frame_candidates = [f for f in frame_candidates if np.abs(f - estimated_frame) < np.min((50, estimated_frame * 0.1))]
        frame_count = int(np.max(frame_candidates)) if len(frame_candidates) > 0 else int(np.ceil(length * fps)) if length and fps else None
    except Exception:
        try:
            cap = cv2.VideoCapture(filename)
            resolution = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > mx_len:
                frame_count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count >= mx_len:
                        raise ValueError(f"Unable to get video properties for video: {filename}")
            if fps == 0:
                fps = 25
            length = frame_count / fps
        except Exception as e:
            raise e
        finally:
            cap.release()
    resolution = [int(x) for x in resolution]
    return *resolution, fps, frame_count, length
