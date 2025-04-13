"""Module for retrieval and ranking system."""
import json
import numpy as np
from config import TOP_K_RESULTS

class RetrievalSystem:
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query, video_id=None, modality=None, limit=TOP_K_RESULTS):
        """Retrieve relevant content based on the query."""
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_text_embedding(query)
        
        # Access the embeddings table
        embeddings_table = self.vector_store.db.open_table("embeddings")
        
        # Build the query
        db_query = embeddings_table.search(query_embedding)
        
        # Filter by video ID if provided
        if video_id:
            db_query = db_query.where(f"video_id = '{video_id}'")
        
        # Filter by modality if provided
        if modality:
            db_query = db_query.where(f"type = '{modality}'")
        
        # Execute the query and get results
        results = db_query.limit(limit).to_list()
        
        # Process results
        processed_results = []
        for result in results:
            metadata = json.loads(result["metadata"])
            processed_result = {
                "id": result["id"],
                "video_id": result["video_id"],
                "type": result["type"],
                "timestamp": result["timestamp"],
                "score": float(result["_distance"]),
                **metadata
            }
            processed_results.append(processed_result)
        
        return processed_results
    
    def retrieve_context_for_query(self, query, video_id):
        """Retrieve comprehensive context for a query about a specific video."""
        # Get the most relevant frames
        frame_results = self.retrieve(
            query=query,
            video_id=video_id,
            modality="frame",
            limit=3
        )
        
        # Get the most relevant subtitle segments
        subtitle_results = self.retrieve(
            query=query,
            video_id=video_id,
            modality="subtitle",
            limit=5
        )
        
        # Combine and sort all results by relevance score
        all_results = frame_results + subtitle_results
        all_results.sort(key=lambda x: x["score"])
        
        # Get the video information
        videos_table = self.vector_store.db.open_table("videos")
        video_info = videos_table.where(f"id = '{video_id}'").to_pandas().iloc[0].to_dict()
        
        # Construct the context information
        context = {
            "video": video_info,
            "frames": frame_results,
            "subtitles": subtitle_results
        }
        
        return context