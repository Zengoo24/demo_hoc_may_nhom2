import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import joblib
from collections import deque
from PIL import Image
import time
from ultralytics import YOLO 

# Thêm khai báo mp_drawing và mp_hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ======================================================================
# I. CẤU HÌNH VÀ HẰNG SỐ CHUNG
# ======================================================================

# --- Cấu hình chung ---
EPS = 1e-8
NEW_WIDTH, NEW_HEIGHT = 640, 480

# --- Cấu hình Drowsiness (Face Mesh) ---
MODEL_PATH = "softmax_model_best1.pkl"
SCALER_PATH = "scale1.pkl"
LABEL_MAP_PATH = "label_map_6cls.json"
SMOOTH_WINDOW = 5
BLINK_THRESHOLD = 0.20
N_FEATURES = 10 

# --- Cấu hình Wheel (Hands) ---
WHEEL_MODEL_PATH = "softmax_wheel_model.pkl"
WHEEL_SCALER_PATH = "scaler_wheel.pkl"
YOLO_MODEL_PATH = "best (1).pt" 
EXPECTED_WHEEL_FEATURES = 128 # Kích thước đặc trưng mới

# ======================================================================
# II. CÁC HÀM TÍNH TOÁN CƠ BẢN VÀ TẢI TÀI NGUYÊN
# ======================================================================

def softmax_predict(X, W, b):
    """Thực hiện dự đoán Softmax (Face Mesh)."""
    logits = X @ W + b
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def softmax_wheel(z):
    """Thực hiện Softmax chuẩn cho Hands/Wheel."""
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

@st.cache_resource
def get_mp_hands_instance():
    """Tạo instance MediaPipe Hands (cho xử lý ảnh tĩnh Vô lăng)."""
    return mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

@st.cache_resource
def load_yolo_model(model_path):
    """Tải mô hình YOLOv8 đã train."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"LỖI TẢI YOLO: Không tìm thấy file {model_path} hoặc lỗi khởi tạo: {e}")
        return None

@st.cache_resource
def load_assets():
    """Tải tất cả tham số mô hình, scaler và label map."""
    try:
        # --- 1. Tải Mô hình Face Mesh ---
        with open(MODEL_PATH, "rb") as f:
            model_data = joblib.load(f)
            W = model_data["W"]
            b = model_data["b"]
        with open(SCALER_PATH, "rb") as f:
            scaler_data = joblib.load(f)
            mean_data = scaler_data["X_mean"]
            std_data = scaler_data["X_std"]
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
            id2label = {int(v): k for k, v in label_map.items()}

        if W.shape[0] != N_FEATURES:
            st.error(f"LỖI KHÔNG TƯƠNG THÍCH: Mô hình FACE MESH yêu cầu {W.shape[0]} đặc trưng, nhưng ứng dụng này trích xuất {N_FEATURES} đặc trưng. Vui lòng kiểm tra lại file model!")
            st.stop()

        # --- 2. Tải Mô hình Wheel/Hands ---
        with open(WHEEL_MODEL_PATH, "rb") as f:
            wheel_model_data = joblib.load(f)
            W_WHEEL = wheel_model_data["W"]
            b_WHEEL = wheel_model_data["b"]
            CLASS_NAMES_WHEEL = wheel_model_data.get("classes", ["off-wheel", "on-wheel"])

        with open(WHEEL_SCALER_PATH, "rb") as f:
            wheel_scaler_data = joblib.load(f)
            X_mean_WHEEL = wheel_scaler_data["X_mean"]
            X_std_WHEEL = wheel_scaler_data["X_std"]
        
        # Kiểm tra kích thước đặc trưng của mô hình Wheel
        if W_WHEEL.shape[0] != EXPECTED_WHEEL_FEATURES:
            st.error(f"LỖI KHÔNG TƯƠNG THÍCH: Mô hình VÔ LĂNG yêu cầu {W_WHEEL.shape[0]} đặc trưng, nhưng code dự kiến {EXPECTED_WHEEL_FEATURES}. Vui lòng kiểm tra và huấn luyện lại mô hình Softmax Vô Lăng.")
            st.stop()

        # --- 3. Tải mô hình YOLOv8 ---
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        if yolo_model is None: 
            st.stop()
            
        # --- 4. Khởi tạo Face Mesh (Global Reference) ---
        mp_face_mesh = mp.solutions.face_mesh
        
        return W, b, mean_data, std_data, id2label, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL, yolo_model

    except FileNotFoundError as e:
        st.error(f"LỖI FILE: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"LỖỖI LOAD DỮ LIỆU: Chi tiết: {e}")
        st.stop()

# Tải tài sản (Chạy một lần)
W, b, mean, std, id2label, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL, YOLO_MODEL = load_assets()
mp_face_mesh = mp.solutions.face_mesh # Global reference

# ======================================================================
# III. HÀM TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT (FACE MESH)
# ======================================================================

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
    return angle_pitch_extra, forehead_y

# ======================================================================
# IV. HÀM TRÍCH XUẤT ĐẶC TRƯNG VÔ LĂNG (WHEEL/HANDS - 128 FEATURES)
# ======================================================================

def detect_wheel_yolo(frame, yolo_model):
    """Phát hiện vô lăng bằng YOLOv8 và trả về (bbox, x, y, r)."""
    # classes=[0] giả định 'steering_wheel' là lớp 0
    results = yolo_model(frame, verbose=False, conf=0.5, classes=[0]) 
    
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = box
            
            x_w = (x_min + x_max) // 2
            y_w = (y_min + y_max) // 2
            r_w = int((x_max - x_min + y_max - y_min) / 4) 
            
            return (x_min, y_min, x_max, y_max), (x_w, y_w, r_w)
            
    return None, None

def extract_wheel_features(image, hands_processor, wheel_coords):
    """
    Trích xuất 128 đặc trưng tay cho mô hình Softmax (Bao gồm tọa độ tuyệt đối,
    khoảng cách tới tâm và các đặc trưng tương đối của đầu ngón tay).
    """
    xw, yw, rw = wheel_coords
    h, w, _ = image.shape
    feats_all = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        # Nếu không tìm thấy tay, trả về None (luật "Không tay = RỜI" sẽ xử lý)
        if not res.multi_hand_landmarks: 
            return None 

        for hand_landmarks in res.multi_hand_landmarks:
            feats = []
            normalized_coords = []
            
            # 1. Trích xuất Tọa độ chuẩn hóa (63 đặc trưng: 21 * x,y,z)
            for lm in hand_landmarks.landmark:
                feats.extend([lm.x, lm.y, lm.z])
                normalized_coords.append(np.array([lm.x, lm.y])) 

            # 2. Đặc trưng Khoảng cách đến tâm vô lăng (1 đặc trưng)
            hx = hand_landmarks.landmark[0].x * w
            hy = hand_landmarks.landmark[0].y * h
            dist_to_center = np.sqrt((xw - hx) ** 2 + (yw - hy) ** 2)
            feats.append(dist_to_center / (rw + EPS))

            # --- THÊM CÁC ĐẶC TRƯNG NÂNG CAO (64 + 10 + 10 = 84 features per hand) ---
            
            # a) Đặc trưng vị trí tương đối của các đầu ngón tay so với tâm vô lăng (10 đặc trưng)
            tip_indices = [4, 8, 12, 16, 20] 
            
            for i in tip_indices:
                lm_tip = hand_landmarks.landmark[i]
                
                tip_x = lm_tip.x * w
                tip_y = lm_tip.y * h
                
                # Khoảng cách tương đối
                rel_dist = np.sqrt((xw - tip_x) ** 2 + (yw - tip_y) ** 2)
                feats.append(rel_dist / (rw + EPS))
                
                # Góc tương đối
                angle = np.arctan2(tip_y - yw, tip_x - xw) / np.pi 
                feats.append(angle)

            # b) Đặc trưng Khoảng cách giữa các ngón tay (10 đặc trưng)
            # (5, 8) = Ngón trỏ, (9, 12) = Ngón giữa, v.v.
            pairs = [(5, 8), (9, 12), (13, 16), (17, 20), (0, 5)] 
            for i, j in pairs:
                p_i = normalized_coords[i]
                p_j = normalized_coords[j]
                
                distance = np.linalg.norm(p_i - p_j)
                feats.append(distance)
                
            feats_all.extend(feats)

        # Đảm bảo đủ độ dài (128) cho mô hình 2 tay
        expected_len = W_WHEEL.shape[0] 
        
        if len(feats_all) < expected_len:
            feats_all.extend([0.0] * (expected_len - len(feats_all)))
        
        feats_all = feats_all[:expected_len]

    return np.array(feats_all, dtype=np.float32)

# ======================================================================
# V. HÀM XỬ LÝ ẢNH TĨNH (WHEEL)
# ======================================================================

def process_static_wheel_image(image_file, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL, YOLO_MODEL):
    img_pil = Image.open(image_file).convert('RGB')
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. PHÁT HIỆN VÔ LĂNG BẰNG YOLO
    bbox_result, wheel_coords = detect_wheel_yolo(img_bgr, YOLO_MODEL)

    if wheel_coords is None:
        cv2.putText(img_bgr, "WHEEL NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "WHEEL NOT FOUND"

    # 2. TRÍCH XUẤT ĐẶC TRƯNG (Sử dụng hàm 128 features mới)
    # Không cần truyền hands_processor vì nó được tạo lại bên trong extract_wheel_features
    features = extract_wheel_features(img_bgr, None, wheel_coords)
    
    final_predicted_class = "off-wheel" 

    # 🛑 LUẬT CỨNG: KHÔNG TAY = RỜI 🛑
    if features is None:
        final_predicted_class = "off-wheel"
        display_label = "ROI"
        final_color = (0, 0, 255) # Đỏ
        text_to_display = "ROI"
    
    else:
        # 3. DỰ ĐOÁN SOFTMAX THUẦN TÚY
        X_sample = features.reshape(1, -1)
        X_scaled = (X_sample - X_mean_WHEEL) / (X_std_WHEEL + EPS)
        z = X_scaled @ W_WHEEL + b_WHEEL
        probabilities = softmax_wheel(z)[0]
        
        predicted_index = np.argmax(probabilities)
        final_predicted_class = CLASS_NAMES_WHEEL[predicted_index]
        confidence = probabilities[predicted_index] * 100
        
        # 4. Gán nhãn hiển thị
        display_label = "CAM" if final_predicted_class == "on-wheel" else "KHONG CAM"
        final_color = (0, 255, 0) if final_predicted_class == "on-wheel" else (0, 0, 255)
        text_to_display = f"{display_label} ({confidence:.1f}%)"
        
        # Vẽ tay (landmarks) lên ảnh BGR
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands_drawer: 
            rgb_for_drawing = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res_for_drawing = hands_drawer.process(rgb_for_drawing)
            if res_for_drawing.multi_hand_landmarks:
                for hand_landmarks in res_for_drawing.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2))

    # 5. Vẽ Vô lăng
    x_min, y_min, x_max, y_max = bbox_result
    xw, yw, rw = wheel_coords
    
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Bounding Box YOLO
    cv2.circle(img_bgr, (xw, yw), rw, (255, 0, 255), 2) # Vòng tròn ước tính từ YOLO
    
    # 6. Đặt text hiển thị cuối cùng
    cv2.putText(img_bgr, text_to_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, final_color, 3, cv2.LINE_AA)
    
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), final_predicted_class.upper()


# ======================================================================
# VII. LỚP XỬ LÝ VIDEO LIVE (WEBRTC PROCESSOR)
# ======================================================================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.W = W; self.b = b; self.mean = mean; self.std = std; self.id2label = id2label
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_pred_label = "CHO DU LIEU VAO"
        self.N_FEATURES = N_FEATURES
        
        self.last_ear_avg = 0.4 
        self.last_pitch = 0.0
        self.pTime = time.time() 
        self.fps = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="bgr24")
        frame_resized = cv2.resize(frame_array, (NEW_WIDTH, NEW_HEIGHT))
        h, w = frame_resized.shape[:2]

        rgb_unflipped = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_unflipped)
        
        ear_avg = 0.0
        delta_ear_value_display = 0.0
        delta_pitch_value_display = 0.0
        
        predicted_label_frame = "NO FACE" 

        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            ear_l = eye_aspect_ratio(landmarks, True)
            ear_r = eye_aspect_ratio(landmarks, False)
            ear_avg = (ear_l + ear_r) / 2.0

            mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y = get_extra_features(landmarks)

            delta_ear_value_display = ear_avg - self.last_ear_avg
            delta_pitch_value_display = pitch - self.last_pitch

            self.last_ear_avg = ear_avg
            self.last_pitch = pitch
            
            if ear_avg < BLINK_THRESHOLD:
                predicted_label_frame = "blink"
            else:
                feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                                  angle_pitch_extra, delta_ear_value_display, forehead_y, delta_pitch_value_display], dtype=np.float32)

                feats_scaled = (feats - self.mean[:self.N_FEATURES]) / (self.std[:self.N_FEATURES] + EPS)
                pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), self.W, self.b)[0]
                predicted_label_frame = self.id2label.get(pred_idx, "UNKNOWN")

            self.pred_queue.append(predicted_label_frame)

        else:
            self.last_ear_avg = 0.4
            self.last_pitch = 0.0
            self.pred_queue.clear() 

        if len(self.pred_queue) > 0:
            self.last_pred_label = max(set(self.pred_queue), key=self.pred_queue.count)
        else:
            self.last_pred_label = "NO FACE"

        cTime = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1 / (cTime - self.pTime + EPS))
        self.pTime = cTime

        frame_display_bgr = frame_resized

        cv2.putText(frame_display_bgr, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display_bgr, f"State: {self.last_pred_label.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

        return av.VideoFrame.from_ndarray(frame_display_bgr, format="bgr24")

# ======================================================================
# VIII. GIAO DIỆN STREAMLIT CHÍNH
# ======================================================================
st.set_page_config(page_title="Demo Softmax", layout="wide")

tab1, tab2 = st.tabs(["🔴 Dự đoán Live Camera", "🚗 Kiểm tra Vô Lăng (Tay)"])

with tab1:
    st.header("1. Nhận diện Trạng thái Khuôn mặt (Live Camera)")
    st.warning("Vui lòng chấp nhận yêu cầu truy cập camera từ trình duyệt của bạn.")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        ICE_SERVERS = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com:3478"]},
            {"urls": ["turn:numb.viagenie.ca:3478"], "username": "webrtc@live.com", "credential": "muazkh"} 
        ]
        
        rtc_config = RTCConfiguration({"iceServers": ICE_SERVERS})

        webrtc_streamer(
            key="softmax_driver_live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with tab2:
    st.header("2. Kiểm tra Vị trí Tay (Vô Lăng)")
    st.warning(f"Nhận diện: {CLASS_NAMES_WHEEL}")
    st.markdown("### Tải lên ảnh tay cầm/rời vô lăng để dự đoán")
    uploaded_wheel_file = st.file_uploader("Chọn một ảnh vô lăng (.jpg, .png)", type=["jpg", "png", "jpeg"], key="wheel_upload")

    if uploaded_wheel_file is not None:
        st.info("Đang xử lý ảnh...")
        # 🛑 ĐÃ SỬA: Truyền YOLO_MODEL vào hàm xử lý ảnh 🛑
        result_img_rgb, predicted_label = process_static_wheel_image(uploaded_wheel_file, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL, YOLO_MODEL)
        st.markdown("---")
        col_img, col_res = st.columns([2, 1])
        with col_img:
            st.image(result_img_rgb, caption="Ảnh đã xử lý (Vô lăng, Tay)", use_container_width=True)
        with col_res:
            st.success("✅ Dự đoán Hoàn tất")
            st.metric(label="Vị trí Tay Dự đoán", value=predicted_label.upper())
            st.caption("Kiểm tra màu sắc: Xanh lá (On-wheel), Đỏ (Off-wheel)")
    else:
        st.info("Vui lòng tải lên một ảnh lên để kiểm tra vị trí tay.")
