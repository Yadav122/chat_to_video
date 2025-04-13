"""Module for LLaVA model integration."""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import os
from dotenv import load_dotenv
from huggingface_hub import login

from config import LLAVA_MODEL

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Login to Hugging Face
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN is missing. Please add it to your .env file.")

class LLMProcessor:
    def __init__(self):
        # Load LLaVA model and processor
        self.processor = AutoProcessor.from_pretrained(LLAVA_MODEL, use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(LLAVA_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def format_prompt(self, query, context):
        """Format prompt with retrieved context for the LLM."""
        system_message = (
            "You are a helpful assistant that answers questions about videos. "
            "Use the provided context information to answer the user's question."
        )

        subtitle_context = ""
        if "subtitles" in context and context["subtitles"]:
            subtitle_context = "Relevant subtitles from the video:\n"
            for i, subtitle in enumerate(context["subtitles"]):
                time_info = f"[{subtitle['start_time']:.2f}s - {subtitle['end_time']:.2f}s]"
                subtitle_context += f"{i+1}. {time_info}: {subtitle['text']}\n"
        
        prompt = f"""
{system_message}

VIDEO INFORMATION:
Title: {context['video']['title']}

{subtitle_context}

USER QUESTION: {query}

Please provide a helpful and accurate answer based on the video content.
"""
        return prompt

    def generate_response(self, query, context, frames_paths=None):
        """Generate response using the LLaVA model."""
        prompt = self.format_prompt(query, context)

        if frames_paths and len(frames_paths) > 0:
            image = Image.open(frames_paths[0]).convert("RGB")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = self.processor.decode(output[0], skip_special_tokens=True)
            return response.strip()
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = self.processor.decode(output[0], skip_special_tokens=True)
            return response.strip()
