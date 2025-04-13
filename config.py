"""Configuration settings for the video chat application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model paths and configurations
#BRIDGETOWER_MODEL = "BridgeTower/bridgetower-large"
BRIDGETOWER_MODEL = "BridgeTower/bridgetower-large-itm-mlm"

#LLAVA_MODEL = os.getenv("LLAVA_MODEL_PATH", "liuhaotian/llava-v1.5-7b")
LLAVA_MODEL = os.getenv("LLAVA_MODEL_PATH", "llava-hf/llava-1.5-7b-hf")

# LanceDB configuration
LANCEDB_URI = str(DATA_DIR / "lancedb")
# HuggingFace Token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Video processing settings
FRAME_EXTRACTION_RATE = 1  # Extract 1 frame per second
MAX_FRAMES = 100  # Maximum number of frames to process

# Retrieval settings
TOP_K_RESULTS = 5  # Number of results to retrieve for each query

