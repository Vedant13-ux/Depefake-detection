import streamlit as st
from model import *


## <- Initial Configiration ->
st.set_page_config(layout="wide", page_title="Deepfake Detection") 
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


## <- Logo ->
st.sidebar.markdown(  """
    <div style = 'position:fixed; left:0; top:0; z-index:100; background: "black"; width: 100%; height: 60px'>
        <img style = "width: 200px; height:60px;z-index:100" src= "https://res.cloudinary.com/ved13/image/upload/v1634567040/logo-500_cxankr.png"></img>
    </div>
""",unsafe_allow_html= True)


## <- Sidebar ->
st.sidebar.markdown("""<br/><br/><br/><hr style="height:1px;border:none;color:gray; background-color:gray; " /> """, unsafe_allow_html=True)
st.sidebar.subheader('Explore the Following')
st.sidebar.write("")

choice_selectbox = st.sidebar.selectbox(
    "Please Select an option",
    ("Introduction", "Deepfake Detection")
)

st.sidebar.write("")

st.sidebar.markdown("""<hr style="height:1px;border:none;color:gray; background-color:gray; " /> """, unsafe_allow_html=True)


## <-Introduction->
if choice_selectbox == "Introduction":
    st.title('Deepfake Detection') 
    st.markdown(f"""
                
        <div style='background: gray;  border-radius: 8px; padding: 15px'>  
            Deepfake techniques, which present realistic AI-generated videos of people doing and saying fictional things, have the potential to have a significant impact on how people determine the legitimacy of information presented online. These content generation and modification technologies may affect the quality of public discourse and the safeguarding of human rights, especially given that deepfakes may be used maliciously as a source of misinformation, manipulation, harassment, and persuasion. Identifying manipulated media is a technically demanding and rapidly evolving challenge that requires collaborations across the entire tech industry and beyond. We have created a Deepfake Detection Model to identify malicious Deepfakes which can be used to mitigate any potential threats. 
        </div>
        <br>
        <center><iframe width="1000" height="600" src="https://www.youtube.com/embed/-QvIX3cY4lc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><center>
        """, unsafe_allow_html= True)


# Deepfake Detection 
if choice_selectbox == "Deepfake Detection":
    st.title('Deepfake Detection')
    col1, col2 = st.columns(2)

    def show_video(video):
        video = video.getvalue()
        return st.video(video, format='video/mp4')
        

    with st.form(key='form'):
        video_upload = st.file_uploader(type=['mp4'], label= 'Upload Deepfake Video (mp4)')
        if video_upload is not None:
            show_video(video_upload)

        submit_button = st.form_submit_button(label='Upload')

        if submit_button:
            if video_upload is not None:
                video_upload_binary = video_upload.getvalue()
                file_name = os.path.join('./videos', f'{video_upload.name}')
                video_file = open(file_name, 'wb')
                video_file.write(video_upload_binary)
                result = predict_on_video(file_name, frames_per_video)
                color = '#EF553B' if result>=0.5 else '#00CC96'
                st.markdown(f'''
                    <div style='background: {color};  border-radius: 8px; padding: 10px; font-size: 18; font-weight: 500'>  
                        Probability of the video being a Deepfake: {result}
                    </div>
                ''', unsafe_allow_html= True)
                st.write('')
                time.sleep(10)
                os.remove(file_name)
                video_upload.close()




    