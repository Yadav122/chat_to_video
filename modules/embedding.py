"""Module for generating embeddings using Hugging Face API."""
import os
import requests
import numpy as np
import base64


class EmbeddingGenerator:
    def __init__(self):
        # Get Hugging Face token from environment variable
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise EnvironmentError("HF_TOKEN not found in environment variables")

        # API headers
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}"
        }

    def _normalize(self, vector):
        """Normalize vector for similarity comparison."""
        return vector / np.linalg.norm(vector)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text using Hugging Face API."""
        api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        payload = {"inputs": text}

        response = requests.post(api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        # The API returns a list of embeddings, we take the first one
        embedding = np.array(response.json())
        return self._normalize(embedding[0])

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embeddings for image using Hugging Face API."""
        try:
            # Open and encode the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        # Use CLIP for image embeddings
        api_url = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
        payload = {"inputs": {"image": image_base64}}

        response = requests.post(api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}: {response.text}")
            return None

        embedding = np.array(response.json())
        return self._normalize(embedding)

    def generate_multimodal_embedding(self, text: str, image_path: str) -> np.ndarray:
        """Generate combined embedding for text and image using Hugging Face API."""
        # For multimodal, we'll use CLIP which handles both text and images
        try:
            # Open and encode the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        api_url = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
        payload = {"inputs": {"text": text, "image": image_base64}}

        response = requests.post(api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}: {response.text}")
            return None

        # For multimodal, we'll average the text and image embeddings
        embedding = np.array(response.json())
        return self._normalize(embedding)

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
