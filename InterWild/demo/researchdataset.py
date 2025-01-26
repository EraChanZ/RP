import cv2
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, frame_skip=1):
        """
        Args:
            video_dir (str): Path to the directory containing video files.
            transform (callable, optional): Optional transform to be applied to each frame.
            frame_skip (int): Number of frames to skip between each loaded frame.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.data, self.metadata = self._load_all_frames_and_metadata()

    def _load_all_frames_and_metadata(self):
        """
        Loads all frames from all videos into memory along with metadata.
        """
        data = []
        metadata = []
        total_files = len(self.video_paths)

        with tqdm(total=total_files, desc="Loading videos") as pbar:  # Create progress bar
            for video_path in self.video_paths:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_name = os.path.basename(video_path)

                frame_num = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video

                    if frame_num % self.frame_skip == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            frame = self.transform(frame)

                        data.append(frame)
                        metadata.append({
                            "video_name": video_name,
                            "frame_num": frame_num,
                            "total_frames": total_frames,
                            "fps": fps
                        })

                    frame_num += 1
                cap.release()
                pbar.update(1)  # Update progress bar after each video is processed

        return data, metadata

    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a frame and its metadata by index.
        """
        return self.data[idx], self.metadata[idx]