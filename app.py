"""Main Streamlit application for the video chat interface."""
import streamlit as st
import os
from pathlib import Path
import time

from modules.video_processor import VideoProcessor
from modules.embedding import EmbeddingGenerator
from modules.indexing import VectorStore
from modules.retrieval import RetrievalSystem
from modules.llm import LLMProcessor

# Initialize the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_title" not in st.session_state:
    st.session_state.video_title = None

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# Initialize components
@st.cache_resource
def load_components():
    video_processor = VideoProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()
    retrieval_system = RetrievalSystem(vector_store, embedding_generator)
    llm_processor = LLMProcessor()
    
    return {
        "video_processor": video_processor,
        "embedding_generator": embedding_generator,
        "vector_store": vector_store,
        "retrieval_system": retrieval_system,
        "llm_processor": llm_processor
    }

components = load_components()

# Application title
st.title("Video Chat Application")

# Sidebar with options
st.sidebar.title("Video Options")

# Video URL input
video_url = st.sidebar.text_input("Enter video URL:")

# Video processing options
include_audio = st.sidebar.checkbox("Include audio", value=True)
include_subtitles = st.sidebar.checkbox("Include subtitles", value=True)

# Process video button
if st.sidebar.button("Process Video"):
    if video_url:
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                # Process the video
                video_processor = components["video_processor"]
                video_data = video_processor.process_video(
                    url=video_url,
                    include_audio=include_audio,
                    include_subtitles=include_subtitles
                )
                
                # Generate embeddings
                embedding_generator = components["embedding_generator"]
                embeddings_data = embedding_generator.process_video_data(video_data)
                
                # Index the video
                vector_store = components["vector_store"]
                index_result = vector_store.index_video(video_url, video_data, embeddings_data)
                
                # Update session state
                st.session_state.video_id = index_result["video_id"]
                st.session_state.video_title = video_data["title"]
                st.session_state.video_processed = True
                st.session_state.video_data = video_data
                
                st.sidebar.success(f"Video processed successfully: {video_data['title']}")
            except Exception as e:
                st.sidebar.error(f"Error processing video: {str(e)}")
    else:
        st.sidebar.error("Please enter a valid video URL")

# Main chat interface
st.subheader("Chat with the Video")

# Display current video information
if st.session_state.video_processed and st.session_state.video_title:
    st.info(f"Current video: {st.session_state.video_title}")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"AI: {message['content']}")

# Chat input
user_query = st.text_input("Ask a question about the video:")

if st.button("Send") and user_query:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    
    # Check if a video has been processed
    if not st.session_state.video_processed:
        response = "Please process a video first before asking questions."
    else:
        with st.spinner("Generating response..."):
            try:
                # Retrieve relevant context
                retrieval_system = components["retrieval_system"]
                context = retrieval_system.retrieve_context_for_query(
                    query=user_query,
                    video_id=st.session_state.video_id
                )
                
                # Get relevant frame paths if available
                frame_paths = None
                if "frames" in context and context["frames"]:
                    frame_paths = [frame["path"] for frame in context["frames"] if "path" in frame]
                
                # Generate response
                llm_processor = components["llm_processor"]
                response = llm_processor.generate_response(
                    query=user_query,
                    context=context,
                    frames_paths=frame_paths
                )
            except Exception as e:
                response = f"Error generating response: {str(e)}"
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
    
    # Rerun to update the display
    st.experimental_rerun()

# Display current video frame if available
if st.session_state.video_processed and "video_data" in st.session_state:
    video_data = st.session_state.video_data
    if "frame_paths" in video_data and video_data["frame_paths"]:
        # Display the first frame
        st.sidebar.subheader("Video Preview")
        st.sidebar.image(str(video_data["frame_paths"][0]))