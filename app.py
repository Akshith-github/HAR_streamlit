import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av,os
import threading,time
from models.lrcnUtil import LRCN

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def clear_output_vids():
    import os
    # list all files with name mp4
    if "output.mp4" in os.listdir():
        os.remove("output.mp4")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(page_title="Streamlit HAR webRTC", page_icon="🚶‍♂️")

task_list = ["Video Stream","videofile"]
st.session_state

with st.sidebar:
    st.title('Task Selection')
    # if st.cache
    # print(st.cache())
    # st.cache
    if 'task_name' in st.session_state:
        task_name = st.selectbox("Select your tasks:", 
                    task_list,index=task_list.index(st.session_state['task_name']))
    else:
        task_name = st.selectbox("select your tasks:",task_list)
    st.session_state['task_name']=task_name

# st.session_state
st.title(task_name)

class VideoProcessor(VideoProcessorBase):
    # model = LRCN().model_Loader()
    # frames = []
    # predictions=[]
    def __init__(self):
        self.model_lock = threading.Lock()
        self.style = style_list[0]

    def update_style(self, new_style):
        if self.style != new_style:
            with self.model_lock:
                self.style = new_style

    def recv(self, frame):
        # img = frame.to_ndarray(format="bgr24")
        img = frame.to_image()
        # print(type(img))
        # self.frames.append(img.copy())
        # print(len(self.frames))
        # if len(self.frames)==21:
        #     self.frames=self.frames[-20:]
            # t1=time.time()

            # print(self.model.predict_single_action_on_frames_list(self.frames))
            # t2 = time.time()
            # print(t2-t1,"seconds")
            # threading.Thread(self.model.predict_single_action_on_frames_list,args=(self.frames,))
        if self.style == style_list[1]:
            img = img.convert("L")
            # print(type(img))

        # return av.VideoFrame.from_ndarray(img, format="bgr24")
        return av.VideoFrame.from_image(img)

if task_name == task_list[0]:
    style_list = ['color', 'black and white']

    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor:

        ctx.video_transformer.update_style(style_selection)
# st.write(dir(st) )
if task_name == task_list[1]:
    a=st.file_uploader("Upload video file for HAR processing", type=["mp4","mkv","avi"])
    # st.file_uploader
    cols = st.columns([3,1,3])
    cols[0].write("## Input video")
    if a:
        with cols[0]:
            st.video(a)
            with st.spinner("Saving Video..."):
                filename = a.name
                with open(filename,"wb") as f:
                    f.writelines(a.readlines())
        model = None
        # model run
        with cols[2]:
            st.write('## Processed Video')
            with st.spinner("Model Analysis ... Running ..."):
                model = LRCN()
                model=model.model_Loader()
                st.write("✅ loaded model")
            clear_output_vids()
            placeholder = st.empty()
            with st.spinner("Running Model Analytics ..."):
                # create tmp file for output if not exists
                # check if tmp folder exists
                if "tmp" not in os.listdir():
                    os.mkdir("tmp")
                # check if output.mp4 exists
                # if outputmp4 exists delete it
                os.remove("tmp\output.mp4") if "output.mp4" in os.listdir("tmp") else None
                for res_tuple in model.predict_on_video(a.name,"tmp\output.mp4"):
                    with placeholder.container():
                        st.write(res_tuple[0])
                        st.image(res_tuple[1])
                placeholder.empty()
                st.write("😇 output is being served..")
                with st.spinner("Serving Output Video ..."):
                    time.sleep(10)
                # video_file = open('tmp\output.mp4', 'rb')
                # video_bytes = video_file.read()
                # video_file.close()
                # st.video(video_bytes)
                video_file = open('tmp\output.mp4', 'rb')
                video_bytes = video_file.read()
                video_file.close()
                # st.video(video_bytes)
                st.download_button("Download Output Video", video_bytes ,"output.mp4")