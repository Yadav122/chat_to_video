"""Module for generating embeddings using BridgeTower."""
import os
import torch
import numpy as np
from PIL import Image
from transformers import BridgeTowerProcessor, BridgeTowerModel

from config import BRIDGETOWER_MODEL  # Make sure this is set to your model name/path


class EmbeddingGenerator:
    def __init__(self):
        # Get Hugging Face token from environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN not found in environment variables")

        # Load BridgeTower model and processor with authentication
        self.processor = BridgeTowerProcessor.from_pretrained(BRIDGETOWER_MODEL, token=hf_token)
        self.model = BridgeTowerModel.from_pretrained(BRIDGETOWER_MODEL, token=hf_token)

        # Set device for inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _normalize(self, vector):
        """Normalize vector for similarity comparison."""
        return vector / np.linalg.norm(vector)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        text_embeddings = outputs.text_embeds.cpu().numpy()
        return self._normalize(text_embeddings[0])

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embeddings for image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        image_embeddings = outputs.image_embeds.cpu().numpy()
        return self._normalize(image_embeddings[0])

    def generate_multimodal_embedding(self, text: str, image_path: str) -> np.ndarray:
        """Generate combined embedding for text and image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        multimodal_embeddings = outputs.pooler_output.cpu().numpy()
        return self._normalize(multimodal_embeddings[0])

    def process_video_data(self, video_data: dict) -> list:
        """
        Process all components of the video and generate embeddings.
        video_data: {
            "frame_paths": [str, str, ...],
            "fps": float,
            "subtitle_segments": [
                {"text": str, "start": float, "end": float}, ...
            ]
        }
        """
        embeddings = []

        # Process frames
        for i, frame_path in enumerate(video_data.get("frame_paths", [])):
            image_embedding = self.generate_image_embedding(frame_path)
            if image_embedding is not None:
                embeddings.append({
                    "embedding": image_embedding,
                    "type": "frame",
                    "frame_id": i,
                    "timestamp": i * (1 / video_data.get("fps", 1)),
                    "path": str(frame_path)
                })

        # Process subtitles
        for i, segment in enumerate(video_data.get("subtitle_segments", [])):
            if isinstance(segment, dict):
                text = segment.get("text", "")
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
            else:
                # Object-style subtitle
                text = getattr(segment, "text", "")
                start_time = getattr(segment, "start", 0)
                end_time = getattr(segment, "end", 0)

            if text.strip():  # Only process non-empty subtitles
                text_embedding = self.generate_text_embedding(text)
                embeddings.append({
                    "embedding": text_embedding,
                    "type": "subtitle",
                    "segment_id": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text
                })

        return embeddings
