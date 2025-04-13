"""Module for downloading and processing videos from URLs."""
import os
import uuid
import cv2
from pytube import YouTube
from pathlib import Path
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

from config import TEMP_DIR, FRAME_EXTRACTION_RATE, MAX_FRAMES

class VideoProcessor:
    def __init__(self, model_size="base"):
        # Load faster-whisper model
        self.whisper_model = WhisperModel(model_size)

    def download_video(self, url):
        try:
            video_id = str(uuid.uuid4())
            output_path = TEMP_DIR / video_id
            output_path.mkdir(exist_ok=True)

            youtube = YouTube(url)
            video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            video_path = video_stream.download(output_path=str(output_path))
            return Path(video_path), youtube.title
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")

    def extract_frames(self, video_path):
        frames = []
        frame_paths = []

        try:
            frames_dir = Path(video_path).parent / "frames"
            frames_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = int(fps * FRAME_EXTRACTION_RATE)

            count = 0
            frame_count = 0

            while cap.isOpened() and frame_count < MAX_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % interval == 0:
                    frame_path = frames_dir / f"frame_{frame_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frames.append(frame)
                    frame_paths.append(frame_path)
                    frame_count += 1

                count += 1

            cap.release()
            return frames, frame_paths
        except Exception as e:
            raise Exception(f"Error extracting frames: {str(e)}")

    def extract_audio(self, video_path):
        try:
            audio_path = Path(video_path).parent / "audio.mp3"
            video_clip = VideoFileClip(str(video_path))
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(str(audio_path))
            audio_clip.close()
            video_clip.close()
            return audio_path
        except Exception as e:
            raise Exception(f"Error extracting audio: {str(e)}")

    def generate_subtitles(self, audio_path):
        try:
            segments, info = self.whisper_model.transcribe(str(audio_path))
            
            subtitle_path = Path(audio_path).parent / "subtitles.txt"
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                segment_list = []
                for segment in segments:
                    f.write(f"{segment.start:.2f} --> {segment.end:.2f}\n{segment.text.strip()}\n\n")
                    segment_list.append(segment)
            
            return subtitle_path, segment_list
        except Exception as e:
            raise Exception(f"Error generating subtitles: {str(e)}")

    def process_video(self, url, include_audio=True, include_subtitles=True):
        try:
            video_path, title = self.download_video(url)
            frames, frame_paths = self.extract_frames(video_path)

            result = {
                "video_path": video_path,
                "title": title,
                "frames": frames,
                "frame_paths": frame_paths,
            }

            if include_audio:
                audio_path = self.extract_audio(video_path)
                result["audio_path"] = audio_path

            if include_subtitles:
                if not include_audio:
                    audio_path = self.extract_audio(video_path)

                subtitle_path, segments = self.generate_subtitles(audio_path)
                result["subtitle_path"] = subtitle_path
                result["subtitle_segments"] = segments

            return result
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")