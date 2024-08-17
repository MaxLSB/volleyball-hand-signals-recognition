import os
import streamlit as st
from io import BytesIO
from backend import ActionDetectionVideo
import requests


def processing_video(uploaded_file, ViewProbabilities, ViewLandmarks):
    video_bytes = BytesIO(uploaded_file.read())
    processed_video = ActionDetectionVideo(video_bytes, ViewProbabilities, ViewLandmarks)
    return processed_video

def download_video(video_url):
    response = requests.get(video_url)
    response.raise_for_status() 
    return response.content  

def main():
    
    st.set_page_config(page_title="Volleyball Signals Recognition ", page_icon="🏐", layout="centered")
    
    st.set_page_config(page_title="Volleyball Signals Recognition ", page_icon="🏐", layout="centered")
    
    st.image('assets/streamlit-banner.png', use_column_width=True)
    st.subheader('VolleyBall Hand Signals Recognition')
    
    col1, col2 = st.columns(2)

    with col1:
        ViewProbabilities = st.checkbox("View Probabilities", value=True)

    with col2:
        ViewLandmarks = st.checkbox("View Landmarks", value=True)
    
    uploaded_file = st.file_uploader("Choose a video file to process...", type=["mp4", "avi", "mov"])
    
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    
    if uploaded_file:
        if st.session_state.processed_video is None:
            status_placeholder = st.empty()
            status_placeholder.warning('Processing your video... (this may take a few seconds)')
            st.session_state.processed_video = processing_video(uploaded_file, ViewProbabilities, ViewLandmarks)
            status_placeholder.success('The video has been processed!')
        
        if st.download_button(
                label="Download Processed Video",
                data=st.session_state.processed_video,
                file_name="processed_video.mp4",
                mime="video/mp4"
            ):
            st.success('The processed video has been downloaded!')
            st.session_state.processed_video = None

    st.write("---")
    
    video_choice = st.selectbox(
        'Select an example video to download (if you don\'t have one):',
        ('Example 1', 'Example 2')
    )
    url_ex2="https://www.dropbox.com/scl/fi/4o58quf3yctc0u2e4d814/example-1.mp4?rlkey=y0uopqydalesz0bzvxzab8rta&st=su594icm&dl=1"
    url_ex1="https://www.dropbox.com/scl/fi/twpo9rwxi5dofxacg0f57/example-2.mp4?rlkey=j98urfyfvf3bz6ntprjzl4wul&st=aqe0d40w&dl=1"
    
    if video_choice == 'Example 1':
        url = url_ex1
        video_data = download_video(url)
    elif video_choice == 'Example 2':
        url = url_ex2
        video_data = download_video(url)

    if st.download_button(
        label="Download Example Video",
        data=video_data,
        file_name="example.mp4",
        mime="video/mp4"
    ):
        st.success('The example video has been downloaded!')

    st.caption('By Maxence Lasbordes')
    
if __name__ == '__main__':
    main()
