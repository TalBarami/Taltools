import cv2
import torch
from torch.utils.data import IterableDataset


class IterableVideoDataset(IterableDataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame
