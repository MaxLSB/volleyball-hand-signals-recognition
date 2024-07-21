import os
import streamlit as st
from io import BytesIO
from backend import ActionDetectionVideo


def processing_video(uploaded_file, ViewProbabilities, ViewLandmarks):
    video_bytes = BytesIO(uploaded_file.read())
    processed_video = ActionDetectionVideo(video_bytes, ViewProbabilities, ViewLandmarks)
    return processed_video

def main():
    
    st.header('VolleyBall Hand Signals Recognition')
    st.caption('By Maxence Lasbordes')
    
    option = st.sidebar.selectbox(
        'Choose an option:',
        ('Use Example Videos', 'Use Your Own Video')
    )
    
    st.markdown(
        """
        <style>
        video {
            max-width: 80%;
            height: auto;
        }
        </style>
        
        """,
        unsafe_allow_html=True
    )
    
    if option == 'Use Example Videos':
        st.subheader("Use Example Videos")
        video_choice = st.selectbox(
            'Choose a video:',
            ('Example 1', 'Example 2')
        )
        file_ex1 = 'examples/example-1.mp4'
        file_ex2 = 'examples/example-2.mp4'
        
        if os.path.isfile(file_ex1) and os.path.isfile(file_ex2):
            if video_choice == 'Example 1':
                video_path = file_ex1
            elif video_choice == 'Example 2':
                video_path = file_ex2
            
            if st.checkbox("View Probabilities"):
                ViewProbabilities = True
            else:
                ViewProbabilities = False
            
            if st.checkbox("View Landmarks"):
                ViewLandmarks = True
            else:
                ViewLandmarks = False
            
            if st.button("Load Video into Model"):
                status_placeholder = st.empty()
                status_placeholder.write('Processing your video... (this may take a few seconds)')
                
                uploaded_file = open(video_path, 'rb')
                processed_video = processing_video(uploaded_file, ViewProbabilities, ViewLandmarks)
                status_placeholder.write('Your video is ready!')
                st.download_button(
                    label="Download Processed Video",
                    data=processed_video,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        
            st.video(video_path)
                      
        else:
            st.write("Please download the example videos with the 'downloadExamples.sh' script !!!")
        
    elif option == 'Use Your Own Video':
        st.subheader("Use Your Own Video")
        
        if st.checkbox("View Probabilities"):
            ViewProbabilities = True
        else:
            ViewProbabilities = False
        
        if st.checkbox("View Landmarks"):
            ViewLandmarks = True
        else:
            ViewLandmarks = False
        
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            status_placeholder = st.empty()
            status_placeholder.write('Processing your video... (this may take a few seconds)')
            processed_video = processing_video(uploaded_file, ViewProbabilities, ViewLandmarks)
            status_placeholder.write('Your video is ready!')
            st.download_button(
                        label="Download Processed Video",
                        data=processed_video,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
        st.image('examples/First.gif')
        st.caption('Exemple of a processed video')
    
if __name__ == '__main__':
    main()