import streamlit as st
from io import BytesIO
from backend import ActionDetectionVideo

def main():
    
    st.title('VolleyBall Hand Signals Recognition')

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        st.write('Processing video...')
        video_bytes = BytesIO(uploaded_file.read())
        
        processed_video = ActionDetectionVideo(video_bytes)

        st.download_button(
                    label="Download Processed Video",
                    data=processed_video,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
    
if __name__ == '__main__':
    main()