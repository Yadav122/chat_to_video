---
title: Video Chat App
emoji: 🎥
colorFrom: blue
colorTo: purple
sdk: "streamlit"
sdk_version: "1.0.0"
app_file: app.py
pinned: false
---

# Video Chat Application

This application allows users to upload YouTube videos and chat with them using AI. The app processes videos to extract frames and subtitles, generates embeddings using the BridgeTower model, and uses LLaVA for generating responses to user queries.

## Features

- Video processing with frame extraction and subtitle generation
- Semantic search using vector embeddings
- Multimodal AI responses using LLaVA
- Interactive chat interface

## Requirements

This application requires:
- A Hugging Face API token for accessing models
- Python 3.8+ and the dependencies listed in requirements.txt

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Hugging Face token: `HF_TOKEN=your_token_here`
4. Run the app: `streamlit run app.py`

