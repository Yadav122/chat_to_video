"""Module for LLaVA model integration using Hugging Face API."""
import os
import requests
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # This loads from .env file if it exists
hf_token = os.getenv("HF_TOKEN")

# Check if token exists
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is missing. Please add it to your environment variables or .env file. If deploying on Render, add HF_TOKEN in the Environment tab of your service.")

class LLMProcessor:
    def __init__(self):
        # Set up API configuration
        self.api_url = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }

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
        """Generate response using the LLaVA model API."""
        prompt = self.format_prompt(query, context)
        payload = {"inputs": prompt}

        # If we have frames, include the first one as an image
        if frames_paths and len(frames_paths) > 0:
            try:
                # Open and encode the image
                with open(frames_paths[0], "rb") as image_file:
                    image_bytes = image_file.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Add image to payload
                payload = {
                    "inputs": {
                        "image": image_base64,
                        "text": prompt
                    }
                }
            except Exception as e:
                print(f"Error processing image {frames_paths[0]}: {e}")
                # Continue with text-only if image fails

        # Make API request
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)

            if response.status_code != 200:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                print(error_msg)
                return f"Error: {error_msg}"

            # Parse the response
            result = response.json()

            # The API returns different formats depending on the model
            if isinstance(result, list):
                return result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)

        except Exception as e:
            error_msg = f"Error calling LLM API: {str(e)}"
            print(error_msg)
            return error_msg
