import streamlit as st
import numpy as np
import joblib
import warnings
import sys
import cv2
import mediapipe as mp
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "softmax_model_best1.pkl"
SCALER_PATH = "scale1.pkl"
SMOOTH_WINDOW = 5
BLINK_THRESHOLD = 0.20
EPS = 1e-8
FPS_SMOOTH = 0.9
N_FEATURES = 10

# ==============================
# LOAD MODEL VÀ SCALER
# ==============================
@st.cache_resource
def load_resources():
    """Tải model, scaler, và khởi tạo MediaPipe FaceMesh một lần."""
    try:
        model_data = joblib.load(MODEL_PATH)
        W = model_data["W"]
        b = model_data["b"]
        CLASSES = model_data["classes"]

        scaler_data = joblib.load(SCALER_PATH)
        X_mean = scaler_data["X_mean"]
        X_std = scaler_data["X_std"]

        idx2label = {i: lbl for i, lbl in enumerate(CLASSES)}

        if W.shape[0] != N_FEATURES:
            st.error(
                f"Lỗi: Kích thước đặc trưng của mô hình ({W.shape[0]}) không khớp với số đặc trưng ({N_FEATURES})."
            )
            sys.exit()

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        return W, b, CLASSES, X_mean, X_std, idx2label, face_mesh

    except FileNotFoundError as e:
        st.error(f"LỖI: Không tìm thấy file tài nguyên: {e}. Vui lòng kiểm tra file .pkl.")
        st.stop()
    except Exception as e:
        st.error(f"LỖI LOAD DỮ LIỆU: {e}")
        st.stop()

# Load tài nguyên
W, b, CLASSES, X_mean, X_std, idx2label, FACE_MESH = load_resources()

# ==============================
# HÀM DỰ ĐOÁN & TÍNH ĐẶC TRƯNG
# ==============================
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z) + EPS)

def predict_proba(x, W, b):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = np.dot(x, W) + b
    return softmax(z)

EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

def eye_aspect_ratio(landmarks, left=True):
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def mouth_aspect_ratio(landmarks):
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def head_pose_yaw_pitch_roll(landmarks):
    left_eye = landmarks[33][:2]
    right_eye = landmarks[263][:2]
    nose = landmarks[1][:2]
    chin = landmarks[152][:2]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx + EPS))
    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))
    baseline = chin - eyes_center
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll

def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist

# ==============================
# VIDEO TRANSFORMER (Lõi xử lý)
# ==============================

class DmsVideoTransformer(VideoTransformerBase):
    
    def __init__(self, W, b, X_mean, X_std, idx2label, blink_thresh, face_mesh_model):
        self.W = W
        self.b = b
        self.X_mean = X_mean
        self.X_std = X_std
        self.idx2label = idx2label
        self.BLINK_THRESHOLD = blink_thresh
        
        self.pTime = time.time()
        self.fps = 0
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_ear_avg = 0.4  
        self.last_pitch = 0.0  
        self.face_mesh = face_mesh_model

        if 'dms_metadata' not in st.session_state:
            st.session_state['dms_metadata'] = {}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        img = cv2.flip(img, 1) 
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)

        final_label = "No Face"
        delta_ear_value = 0.0
        delta_pitch_value = 0.0
        ear_avg = 0.0
        
        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])
            
            # 1. TÍNH TOÁN CÁC ĐẶC TRƯNG TĨNH
            ear_l = eye_aspect_ratio(landmarks, True)
            ear_r = eye_aspect_ratio(landmarks, False)
            mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, _ = get_extra_features(landmarks)

            # 2. TÍNH ĐẶC TRƯNG ĐỘNG 
            ear_avg = (ear_l + ear_r) / 2.0
            delta_ear_value = ear_avg - self.last_ear_avg
            delta_pitch_value = pitch - self.last_pitch

            # 3. CẬP NHẬT LỊCH SỬ
            self.last_ear_avg = ear_avg
            self.last_pitch = pitch
            
            # 4. ÁP DỤNG LUẬT CỨNG (HEURISTIC) CHO BLINK
            current_pred_label = "unknown"
            
            if ear_avg < self.BLINK_THRESHOLD:
                current_pred_label = "blink"
            else:
                feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                                angle_pitch_extra, delta_ear_value, forehead_y, delta_pitch_value], dtype=np.float32)

                # CHUẨN HÓA & DỰ ĐOÁN
                feats_scaled = (feats - self.X_mean[:N_FEATURES]) / (self.X_std[:N_FEATURES] + EPS)
                probs = predict_proba(feats_scaled, self.W, self.b)
                pred_idx = np.argmax(probs)
                current_pred_label = self.idx2label[pred_idx]

            # LÀM MƯỢT KẾT QUẢ
            self.pred_queue.append(current_pred_label)
            final_label = max(set(self.pred_queue), key=self.pred_queue.count)
            
            # Cập nhật metadata
            st.session_state['dms_metadata'] = {
                'final_label': final_label,
                'ear_avg': ear_avg,
                'delta_ear': delta_ear_value,
                'delta_pitch': delta_pitch_value,
                'feats': feats.tolist()
            }

        else:
            self.last_ear_avg = 0.4
            self.last_pitch = 0.0
            self.pred_queue.clear()
            st.session_state['dms_metadata'] = {'final_label': 'No Face', 'ear_avg': 0.0, 'delta_ear': 0.0, 'delta_pitch': 0.0, 'feats': [0.0]*N_FEATURES}


        # Tính và hiển thị FPS
        cTime = time.time()
        self.fps = FPS_SMOOTH * self.fps + (1 - FPS_SMOOTH) * (1 / (cTime - self.pTime + EPS))
        self.pTime = cTime
        
        # HIỂN THỊ TRÊN KHUNG HÌNH
        color = (0, 255, 0) 
        if final_label.lower() == "blink":
            color = (0, 0, 255) 
        elif final_label.lower() == "nod":
            color = (0, 255, 255) 
            
        cv2.putText(img, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"State: {final_label.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        cv2.putText(img, f"EAR: {ear_avg:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Delta Pitch: {delta_pitch_value:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return img

# ==============================
# GIAO DIỆN STREAMLIT
# ==============================

st.set_page_config(
    page_title="DMS Softmax - Camera Test",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📹 DMS: Softmax Model Camera Test (WebRTC)")
st.markdown("---")

FEATURE_DESC = {
    0: "EAR_L", 1: "EAR_R", 2: "MAR", 3: "YAW", 4: "PITCH", 
    5: "ROLL", 6: "ANGLE_PITCH_EXTRA", 7: "DELTA_EAR", 
    8: "FOREHEAD_Y", 9: "DELTA_PITCH",
}

with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.subheader("Tham số Heuristic")
    thresh = st.slider("BLINK_THRESHOLD", 0.05, 0.40, BLINK_THRESHOLD, 0.01)

    st.subheader("Trạng thái Mô hình")
    st.write(f"Classes: {CLASSES}")
    st.write(f"Features: {N_FEATURES}")
    st.write(f"Smoothing Window: {SMOOTH_WINDOW}")
    st.write("---")
    st.markdown("⚠️ **Lưu ý:** Ứng dụng cần quyền truy cập camera.")

col_cam, col_data = st.columns([2, 1])

with col_cam:
    st.subheader("Camera Trực tiếp & Phân tích")
    
    # BỘ ICE SERVERS MỞ RỘNG (ĐÃ SỬA)
    ICE_SERVERS = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]},
        # Thêm một TURN server công khai (dễ bị quá tải)
        {"urls": ["turn:numb.viagenie.ca:3478"], "username": "webrtc@live.com", "credential": "muazkh"} 
    ]
    
    webrtc_ctx = webrtc_streamer(
        key="dms-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": ICE_SERVERS}), 
        video_processor_factory=lambda: DmsVideoTransformer(W, b, X_mean, X_std, idx2label, thresh, FACE_MESH),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_data:
    st.subheader("Kết quả Phân tích (Cập nhật)")
    
    status_placeholder = st.empty()
    feature_placeholder = st.empty()

    if webrtc_ctx.state.playing:
        while webrtc_ctx.state.playing:
            metadata = st.session_state.get('dms_metadata', {})

            if metadata and metadata['final_label'] != 'No Face':
                final_label = metadata.get('final_label', 'UNKNOWN')
                color_map = {"blink": "red", "nod": "darkorange", "yawn": "blue", "smile": "green", "unknown": "gray", "no face": "gray"}
                
                status_html = f"""
                <div style='background-color: {color_map.get(final_label.lower(), 'purple')}; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h1 style='color: white; margin: 0;'>{final_label.upper()}</h1>
                </div>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                feats = metadata.get('feats', [0.0]*N_FEATURES)
                feature_names = [FEATURE_DESC[i] for i in range(N_FEATURES)]
                
                feature_data = {
                    "Đặc trưng": feature_names,
                    "Giá trị": [f"{f:.4f}" for f in feats]
                }
                feature_placeholder.dataframe(feature_data, use_container_width=True, hide_index=True)

            else:
                status_placeholder.warning("🔴 Đang chờ khuôn mặt hoặc camera chưa bật. Nhấn START.")
                feature_placeholder.empty()

            time.sleep(0.1)
    else:
        st.info("Nhấn 'START' để bắt đầu camera và phân tích. Vui lòng cấp quyền truy cập camera.")
