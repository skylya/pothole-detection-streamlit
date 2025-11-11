# ============================================================
# Automated Pothole Detection System - Streamlit Interactive App
# ============================================================
import streamlit as st
import tempfile, os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2 

# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(page_title="Automated Pothole Detection", layout="wide")
st.title("🕳️ Automated Pothole Detection System")
st.markdown("""
Upload **images** or **videos** of roads to automatically detect potholes using a pretrained **YOLOv8 model**.  
You can preview detections and download the processed results.
""")

# ============================================================
# LOAD YOLO MODEL
# ============================================================
@st.cache_resource
def load_model():
    model_path = "best.pt"  
    model = YOLO(model_path)
    return model

model = load_model()
st.sidebar.success("YOLOv8 model loaded successfully!")

# ============================================================
# FILE UPLOAD SECTION
# ============================================================
st.header("📁 Upload Road Image or Video")

uploaded_file = st.file_uploader(
    "Upload an image or video file for pothole detection",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    help="Upload road footage or an image containing potholes"
)

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # ============================================================
    # IMAGE DETECTION
    # ============================================================
    if file_ext in ["jpg", "jpeg", "png"]:
        st.subheader("Image Detection Preview")

        results = model.predict(source=input_path, conf=0.25)
        result = results[0].plot()  # Annotated frame as NumPy array (BGR)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        st.image(result_rgb, caption="Detected Potholes", use_container_width=True)

        output_path = os.path.join(temp_dir, "detected_image.jpg")
        cv2.imwrite(output_path, result)

        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Image",
                data=file,
                file_name="pothole_detected.jpg",
                mime="image/jpeg"
            )

    # ============================================================
    # VIDEO DETECTION
    # ============================================================
    elif file_ext in ["mp4", "mov", "avi"]:
        st.subheader("🎥 Video Detection Processing")
        st.info("Processing video frame by frame to prevent memory overflow")

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(temp_dir, "detected_video.mp4")
    
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
        # Process frames in stream mode to save RAM
        results = model(source=input_path, stream=True, conf=0.25)
    
        for r in results:
            frame = r.plot()  # Draw YOLO detections on frame
            out.write(frame)
    
        cap.release()
        out.release()
    
        st.success("Video processed successfully!")
        st.video(output_path)
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="pothole_detected.mp4",
                mime="video/mp4"
            )

        # --- Free up memory ---
        import gc, torch
        torch.cuda.empty_cache()
        gc.collect()

# ============================================================
# SIDEBAR INFO
# ============================================================
st.sidebar.header("⚙️ Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
st.sidebar.write("Lower threshold → More detections, but possibly less accurate.")

st.sidebar.header("📊 Model Info")
st.sidebar.write("- **Model:** YOLOv8s (pretrained)")
st.sidebar.write("- **Framework:** Ultralytics YOLO")
st.sidebar.write("- **Purpose:** Pothole severity detection and localization")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<center>Developed for Automated Road Condition Monitoring using Deep Learning 🚗</center>",
    unsafe_allow_html=True
)



