import sys
sys.path.append(r"..\CCIT4080A\ml")
sys.path.append(r"..\CCIT4080A")
import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (webrtc_streamer, WebRtcMode)
from ml.Movenet import Movenet
from ml.Classifier import Classifier
from ml.Draw_predict import Draw_predict
from ml.Add_html import Add_html
import queue
import time

st.set_page_config(page_title="Plank Estimation", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("Plank Estimation")
st.header("", divider="red")
with st.sidebar:
    st.image("For_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1,1,1,1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chen")
    st.header("", divider="red")

model_name = st.selectbox("Movenet Model Select:", (
    "Movenet lightning (float 16)",
    "Movenet thunder (float 16)",
    "Movenet lightning (int 8)",
    "Movenet thunder (int 8)"))
th1 = st.slider("confidence threshold", 0.0, 1.0, 0.3, 0.05)
st.caption("Suggest the confidence threshold should be setted between 0.3 to 0.4 to get the best result")

movenet = Movenet(model_name)
classify = Classifier("pose_classifier.tflite", "pose_labels.txt")
draw_predict = Draw_predict()
add_html = Add_html()
Error = add_html.autoplay_audio("Error.mp3")
popup = add_html.popup_window()
st.markdown(Error,unsafe_allow_html=True)
KEYPOINT = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
label_queue = queue.Queue()
output_queue = queue.Queue()
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    keypoints_with_scores = movenet.movenet(image)
    x, y, c = image.shape
    pose_class_names, output = classify.classtify(keypoints_with_scores)
    if output[0] >= output[1]:
        output_label = pose_class_names[0]
    else:
        output_label = pose_class_names[1]
    output_queue.put(output)
    label_queue.put(output_label)
    draw_predict.draw_connections(image, keypoints_with_scores, th1)
    draw_predict.draw_keypoints(image, keypoints_with_scores, th1)
    keypoints_with_scores = np.multiply(keypoints_with_scores, [x, y, 1])
    result_queue.put(keypoints_with_scores)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webRTC =webrtc_streamer(key="Pose Detection",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

label_predict = st.empty()
label_msg = st.empty()
keypoint_message = st.empty()
fps_message = st.empty()
if webRTC.state.playing:
    start_time = time.time()
    continue_time = 0
    counter = 0
    while True:
        label = label_queue.get()
        if label == "non_standard":
            continue_time = time.time()

        output = output_queue.get()
        counter += 1
        fps =counter // (time.time()-start_time)
        check_pose = pd.DataFrame({"Non Standard": [str(round(output[0] * 100, 2)) + "%"],
                                   "Standard": [str(round(output[1] * 100, 2)) + "%"]}, index=["prediction"])
        label_predict.table(check_pose)
        label_msg.write(label)
        result = result_queue.get()
        result = np.squeeze(result)
        more_result = pd.DataFrame({
            "Keypoints": KEYPOINT,
            "X Coordinate": result[:, 1],
            "Y Coordinate": result[:, 0],
            "confidence threshold": result[:, 2]
        }, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        keypoint_message.table(more_result)
        fps_message.write(f"Fps = {fps}")
        counter = 0
        start_time = time.time()



st.markdown("This demo uses a model and code from")
st.markdown("https://tfhub.dev/google/movenet/singlepose/lightning/4")
st.markdown("https://tfhub.dev/google/movenet/singlepose/thunder/4")
st.markdown("https://tensorflow.google.cn/lite/tutorials/pose_classification?hl=zh-cn")
st.markdown("Many thanks to the project")