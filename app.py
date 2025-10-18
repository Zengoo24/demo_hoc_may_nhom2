import streamlit as st
import numpy as np
import joblib
from collections import deque
import warnings
import sys

# ==============================
# CẤU HÌNH (Sử dụng lại từ code gốc)
# ==============================
MODEL_PATH = "softmax_model_best1.pkl"
SCALER_PATH = "scale1.pkl"
SMOOTH_WINDOW = 5
BLINK_THRESHOLD = 0.20
EPS = 1e-8
N_FEATURES = 10  # SỐ LƯỢNG ĐẶC TRƯNG MONG ĐỢI: 10

# ==============================
# LOAD MODEL VÀ SCALER
# ==============================
@st.cache_resource
def load_resources():
    """Tải model và scaler, dùng st.cache_resource để chỉ tải 1 lần."""
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

        return W, b, CLASSES, X_mean, X_std, idx2label

    except FileNotFoundError as e:
        st.error(f"LỖI: Không tìm thấy file tài nguyên: {e}. Vui lòng kiểm tra file .pkl.")
        st.stop()
    except Exception as e:
        st.error(f"LỖI LOAD DỮ LIỆU: {e}")
        st.stop()


# Load tài nguyên
W, b, CLASSES, X_mean, X_std, idx2label = load_resources()

# ==============================
# HÀM DỰ ĐOÁN (Sử dụng lại từ code gốc)
# ==============================
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z) + EPS)

def predict_proba(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = np.dot(x, W) + b
    return softmax(z)

# ==============================
# ỨNG DỤNG STREAMLIT
# ==============================
st.set_page_config(
    page_title="DMS Softmax Mock-up (10 Feats)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ Hệ thống Giám sát Tài xế (DMS) - Softmax Mock-up")
st.markdown("---")

# Dictionary mô tả 10 đặc trưng
FEATURE_DESC = {
    0: "EAR_L (Eye Aspect Ratio - Left)",
    1: "EAR_R (Eye Aspect Ratio - Right)",
    2: "MAR (Mouth Aspect Ratio)",
    3: "YAW (Quay ngang đầu)",
    4: "PITCH (Cúi/Ngửa đầu)",
    5: "ROLL (Nghiêng đầu)",
    6: "ANGLE_PITCH_EXTRA (Góc Ngửa/Cúi - Dựa trên Chin/Nose Z)",
    7: "DELTA_EAR (Thay đổi EAR)",
    8: "FOREHEAD_Y (Tọa độ Y trán)",
    9: "DELTA_PITCH (Thay đổi PITCH - Cho Nod)",
}

# --- SIDEBAR: CẤU HÌNH VÀ SIMULATION ---
with st.sidebar:
    st.header("⚙️ Cấu hình & Giả lập")
    st.subheader("Tham số Heuristic")
    thresh = st.slider("BLINK_THRESHOLD", 0.05, 0.40, BLINK_THRESHOLD, 0.01)

    st.subheader("Nhập Đặc trưng Thủ công")
    st.markdown("⚠️ *Nhập 10 giá trị đặc trưng để kiểm tra mô hình.*")

    # Tạo các slider/number_input cho 10 đặc trưng
    features_input = {}
    default_vals = [0.3, 0.3, 0.05, 0.0, 0.0, 0.0, 90.0, 0.0, 200.0, 0.0]
    ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0), (60.0, 120.0), (-0.2, 0.2), (100.0, 400.0), (-0.2, 0.2)]

    for i in range(N_FEATURES):
        key = FEATURE_DESC[i].split(' ')[0]
        min_v, max_v = ranges[i]
        features_input[i] = st.number_input(
            f"{i+1}. {FEATURE_DESC[i]} ({key})",
            value=float(default_vals[i]),
            min_value=min_v,
            max_value=max_v,
            step=0.01 if ranges[i][1] < 1 else 1.0,
            format="%.4f"
        )

    feats = np.array([features_input[i] for i in range(N_FEATURES)], dtype=np.float32)


# --- MAIN CONTENT: HIỂN THỊ KẾT QUẢ ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Giá trị Đặc trưng Hiện tại")
    feature_data = {
        "Đặc trưng": [FEATURE_DESC[i] for i in range(N_FEATURES)],
        "Giá trị": [f"{f:.4f}" for f in feats]
    }
    st.dataframe(feature_data, use_container_width=True, hide_index=True)


with col2:
    st.subheader("🧠 Kết quả Dự đoán")

    ear_avg = (feats[0] + feats[1]) / 2.0
    current_pred_label = "unknown"

    # 1. ÁP DỤNG LUẬT CỨNG (HEURISTIC) CHO BLINK
    st.write(f"**Trung bình EAR (L+R)/2:** {ear_avg:.4f}")
    if ear_avg < thresh:
        current_pred_label = "blink"
        st.warning(f"🚨 **PHÁT HIỆN BLINK** (EAR < {thresh})")
        st.markdown(f"**Phán đoán Cuối cùng:** <span style='font-size: 30px; color: yellow; background-color: darkred; padding: 5px; border-radius: 5px;'>{current_pred_label.upper()}</span>", unsafe_allow_html=True)
    else:
        # 2. Dùng Softmax cho các hành vi khác

        st.info("💡 Áp dụng Softmax...")
        
        # CHUẨN HÓA
        feats_scaled = (feats - X_mean[:N_FEATURES]) / (X_std[:N_FEATURES] + EPS)

        # DỰ ĐOÁN Softmax
        probs = predict_proba(feats_scaled)
        probs = np.array(probs).flatten()
        pred_idx = np.argmax(probs)
        current_pred_label = idx2label[pred_idx]

        # Hiển thị kết quả Softmax
        color_map = {"nod": "green", "yawn": "orange", "smile": "blue", "unknown": "gray"}
        
        st.markdown(f"**Phán đoán Softmax:** <span style='font-size: 30px; color: white; background-color: {color_map.get(current_pred_label, 'purple')}; padding: 5px; border-radius: 5px;'>{current_pred_label.upper()}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Chi tiết Xác suất Softmax")
        prob_data = {
            "Hành vi": CLASSES,
            "Xác suất": [f"{p*100:.2f}%" for p in probs]
        }
        st.table(prob_data)

st.markdown("---")
st.subheader("Tham số Mô hình")
col_model1, col_model2 = st.columns(2)
col_model1.metric("Số lớp", len(CLASSES))
col_model2.metric("Số đặc trưng (W.shape[0])", W.shape[0])

# Cảnh báo về tính năng "Làm mượt"
st.warning("⚠️ **Lưu ý:** Chức năng **Làm mượt (Smoothing)** bằng `pred_queue` (deque) không được mô phỏng trong giao diện tĩnh này vì nó yêu cầu một luồng dữ liệu thời gian thực và lịch sử.")
