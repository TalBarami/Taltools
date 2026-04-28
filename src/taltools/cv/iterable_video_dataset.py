import cv2

try:
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError("Install taltools[torch] to use IterableVideoDataset") from e


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
