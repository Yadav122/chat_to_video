"""Module for storing embeddings in LanceDB."""
import lancedb
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pyarrow as pa  # Import pyarrow for schema definitions

from config import LANCEDB_URI

class VectorStore:
    def __init__(self):
        # Initialize LanceDB
        self.db = lancedb.connect(LANCEDB_URI)
        
        # Create or get tables
        self._initialize_tables()
    
    def _initialize_tables(self):
     """Initialize LanceDB tables if they don't exist."""
     if "videos" not in self.db.table_names():
        schema = pa.schema([
            ("id", pa.string()),
            ("title", pa.string()),
            ("url", pa.string()),
            ("created_at", pa.timestamp("ns"))
        ])
        self.db.create_table("videos", schema=schema)
    
     if "embeddings" not in self.db.table_names():
        schema = pa.schema([
            ("id", pa.string()),
            ("video_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), list_size=768)),  # ✅ fixed!
            ("type", pa.string()),
            ("metadata", pa.string()),
            ("timestamp", pa.float32())
        ])
        self.db.create_table("embeddings", schema=schema)

    
    def add_video(self, url, title, video_id):
        """Add video metadata to the database."""
        videos_table = self.db.open_table("videos")
        
        # Create video metadata record
        video_data = {
            "id": video_id,
            "title": title,
            "url": url,
            "created_at": pd.Timestamp.now()
        }
        
        # Add to table
        videos_table.add(pd.DataFrame([video_data]))
        return video_id
    
    def add_embeddings(self, video_id, embeddings_data):
        """Add embeddings to the database."""
        embeddings_table = self.db.open_table("embeddings")
        
        # Prepare data for insertion
        records = []
        
        for i, data in enumerate(embeddings_data):
            embedding = data["embedding"]
            
            # Remove embedding from metadata
            metadata = data.copy()
            del metadata["embedding"]
            
            record = {
                "id": f"{video_id}_{i}",
                "video_id": video_id,
                "embedding": embedding,
                "type": data["type"],
                "metadata": json.dumps(metadata),
                "timestamp": metadata.get("timestamp", 0) if "timestamp" in metadata else (
                    metadata.get("start_time", 0) if "start_time" in metadata else 0
                )
            }
            
            records.append(record)
        
        # Add to table
        embeddings_table.add(pd.DataFrame(records))
        return len(records)
    
    def index_video(self, url, video_data, embeddings_data):
        """Index a processed video with its embeddings."""
        # Add video metadata
        video_id = Path(video_data["video_path"]).parent.name  # Use the unique folder name as ID
        self.add_video(url, video_data["title"], video_id)
        
        # Add embeddings
        num_embeddings = self.add_embeddings(video_id, embeddings_data)
        
        return {
            "video_id": video_id,
            "num_embeddings": num_embeddings
        }
