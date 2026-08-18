import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict

def main():
    st.title("Image Classify Project from YOLO")
    st.write('<p style="font-size: 22px;">ผลการจัดจำแนกรูปภาพ Image classify from YOLO Model</p>', unsafe_allow_html=True)
    st.sidebar.header("งานวิจัยของศุภกร วงษ์เรืองพิบูล")

    model_path = "BestImageClassify.pt"
    uploaded_file = st.sidebar.file_uploader("Please select image file…", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
            tfile.write(uploaded_file.read())
            tmp_path = tfile.name
        model = YOLO(model_path)
        img = cv2.imread(tmp_path)
        st.sidebar.info("Image Classifying ...")
        results = model(tfile.name, imgsz=640)[0]
        
        if results.probs is not None and hasattr(results, 'names'):
            predicted_class_index = int(results.probs.data.argmax())
            predicted_class = results.names[predicted_class_index]
            confidence = float(results.probs.data.max()) * 100
        else:
            predicted_class = 'Unknown'
            confidence = 0.0

        img_with_boxes = results.plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, width=640, caption=f'ผลการจัดจำแนกอันดับที่ 1: {predicted_class} (ความมั่นใจ {confidence:.2f}%)')

        try:
            tfile.close()
            os.unlink(tfile.name)
        except Exception as e:
            st.error(f" An error occurred while deleting temporary files: {e}")

main()
