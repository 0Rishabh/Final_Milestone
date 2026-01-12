
# Hand Gesture Volume Control System



import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
from pycaw.pycaw import AudioUtilities
import time


st.set_page_config(
    page_title="Hand Gesture Volume Control",
    page_icon="✋",
    layout="wide"
)


st.markdown("""
<style>
.card {
    background-color: pink;
    padding: 0px 2px;
    border-radius: 40px;
    margin-bottom: -29px;
    box-shadow: 3px 10px 16px rgba(0,0,0,0.15);
    text-align: center;
    transform: scale(0.70);
}
.card h4 {
    margin-bottom: 14px 14px;
    color: #555;
}
.card h2 {
    margin: 2px;
    color: #000;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("🎛 Controls")
run = st.sidebar.toggle("Start Camera", value=False)
st.sidebar.markdown("**Gesture:** Thumb + Finger")
st.sidebar.markdown("- Index / Middle / Ring / Pinky")


st.markdown("<h1 style='text-align:center;'>✋ Hand Gesture Volume Control</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Control system volume using hand gestures</p>", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])
frame_window = col1.image([], use_container_width=True)

with col2:
    st.subheader("📊 Status")
    finger_box = st.empty()
    volume_box = st.empty()
    accuracy_box = st.empty()
    distance_box = st.empty()
    volume_bar = st.progress(0)


volume = AudioUtilities.GetSpeakers().EndpointVolume
vol_min, vol_max = volume.GetVolumeRange()[:2]


smooth = deque(maxlen=5)
def smooth_val(v):
    smooth.append(v)
    return int(sum(smooth) / len(smooth))


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

finger_map = {
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20
}


def calculate_accuracy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    noise = np.std(gray)

    brightness_score = np.interp(brightness, [50, 200], [60, 100])
    noise_penalty = np.interp(noise, [10, 60], [0, 30])

    accuracy = brightness_score - noise_penalty
    return int(np.clip(accuracy, 40, 100))


cap = cv2.VideoCapture(0)
PIXEL_TO_MM = 0.26 

while run:
    ok, frame = cap.read()
    if not ok:
        st.error("❌ Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    active_finger = "None"
    vol_percent = 0
    distance_mm = 0

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
            thumb_x, thumb_y = lm[4]

            for name, idx in finger_map.items():
                fx, fy = lm[idx]
                base_y = lm[idx - 2][1]

                if fy < base_y:
                    active_finger = name
                    raw_dist = math.hypot(fx - thumb_x, fy - thumb_y)
                    dist = smooth_val(raw_dist)

                    distance_mm = int(dist * PIXEL_TO_MM)

                    MAX_OPEN_DISTANCE = 180
                    vol = np.interp(dist, [30, MAX_OPEN_DISTANCE], [vol_min, vol_max])
                    vol_percent = int(np.interp(dist, [30, MAX_OPEN_DISTANCE], [0, 100]))

                    volume.SetMasterVolumeLevel(vol, None)

                    cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), -1)
                    cv2.circle(frame, (fx, fy), 8, (255, 0, 255), -1)
                    cv2.line(frame, (thumb_x, thumb_y), (fx, fy), (255, 0, 255), 2)
                    break

            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    accuracy = calculate_accuracy(frame)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    finger_box.markdown(f"""
    <div class="card">
        <h4>Active Finger</h4>
        <h2>{active_finger}</h2>
    </div>
    """, unsafe_allow_html=True)

    volume_box.markdown(f"""
    <div class="card">
        <h4>Volume</h4>
        <h2>{vol_percent}%</h2>
    </div>
    """, unsafe_allow_html=True)

    accuracy_box.markdown(f"""
    <div class="card">
        <h4>Accuracy</h4>
        <h2>{accuracy}%</h2>
    </div>
    """, unsafe_allow_html=True)

    distance_box.markdown(f"""
    <div class="card">
        <h4>Distance</h4>
        <h2>{distance_mm} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    volume_bar.progress(min(vol_percent, 100))
    time.sleep(0.01)

cap.release()