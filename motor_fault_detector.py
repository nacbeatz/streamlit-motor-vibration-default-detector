import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ------------------- Streamlit UI Setup ---------------------
st.set_page_config(page_title="Motor Vibration Fault Detector", layout="wide")
st.title("🔧 Motor Vibration Fault Detection System")
st.markdown("This app detects abnormal vibration patterns in motors using video input.")

# ------------------- User Input Controls --------------------
video_source = st.radio("Choose input source:", ("Upload Video", "Use Live Camera"))
threshold = st.slider("Vibration Threshold", 0.0, 10.0, 1.0, step=0.1)

uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload motor video", type=["mp4", "avi"])

# ------------------- Helper Function ------------------------
def analyze_frame_motion(prev_gray, gray, prev_pts):
    # Compute optical flow (movement) between two frames
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    good_new = new_pts[status == 1]
    good_old = prev_pts[status == 1]
    movements = np.linalg.norm(good_new - good_old, axis=1)
    return movements, good_new.reshape(-1, 1, 2)

# ------------------- Video Processing -----------------------
def process_video(cap):
    vibration_magnitudes = []
    motion_exceeded = []

    ret, old_frame = cap.read()
    if not ret:
        st.error("❌ Failed to read video. Please check your source.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movements, prev_pts = analyze_frame_motion(old_gray, gray, prev_pts)

        vibration_level = np.mean(movements)
        vibration_magnitudes.append(vibration_level)
        motion_exceeded.append(vibration_level > threshold)

        # Show live vibration level
        st.write(f"📊 Frame {frame_count}: Vibration Magnitude = {vibration_level:.2f}")
        frame_count += 1

        old_gray = gray.copy()

    cap.release()

    # ----------------- Analysis Results -------------------
    st.subheader("📉 Vibration Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(vibration_magnitudes, label="Vibration Magnitude")
    ax1.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # ----------------- Frequency Domain -------------------
    st.subheader("🔊 Frequency Spectrum")
    yf = rfft(vibration_magnitudes)
    xf = rfftfreq(len(vibration_magnitudes), 1)  # assume 1 frame per time unit
    fig2, ax2 = plt.subplots()
    ax2.plot(xf, np.abs(yf))
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    st.pyplot(fig2)

    # ----------------- Fault Detection -------------------
    abnormal_percent = (np.sum(motion_exceeded) / len(motion_exceeded)) * 100
    if abnormal_percent > 15:
        st.error("⚠️ Abnormal vibration detected! Potential motor fault.")
        st.markdown("**Possible Causes:**\n- Loose components\n- Unbalanced rotation\n- Shaft misalignment\n- Bearing wear\n- Resonance issues")
    else:
        st.success("✅ Vibration levels are within normal range.")

# ------------------- Run App Based on Input -----------------

if video_source == "Upload Video" and uploaded_file is not None:
    st.video(uploaded_file)
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture("temp_video.mp4")
    process_video(cap)

elif video_source == "Use Live Camera":
    st.warning("📷 Live camera will start. Press 'Stop' to end.")
    try:
        cap = cv2.VideoCapture(0)
        process_video(cap)
    except Exception as e:
        st.error(f"Failed to start webcam: {e}")
