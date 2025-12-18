# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
from pathlib import Path
import timm
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# ============================================
# MODEL DEFINITION (Copy dari notebook mobile)
# ============================================
class MobileDrowsinessModel(nn.Module):
    """
    Mobile-Optimized Architecture for Drowsiness Detection
    """
    def __init__(self, num_classes=4, num_frames=8, dropout=0.2):
        super().__init__()
        self.num_frames = num_frames
        
        # Backbone: MobileNetV3-Small (lightweight)
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
        
        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 112, 112)
            dummy_feat = self.backbone(dummy)
        self.feature_dim = dummy_feat.shape[1]
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.lstm_feature_dim = 128
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        
        # Extract CNN features per frame
        x_flat = x.view(batch_size * num_frames, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_features = h_n[-1]
        
        # Classification
        output = self.classifier(final_features)
        return output


# ============================================
# AUDIO WARNING UTILITIES
# ============================================
def autoplay_audio(file_path):
    """
    Auto-play audio file using HTML5 audio tag
    """
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)


def play_warning_sound(audio_path=None, warning_type="yawning"):
    """
    Play warning sound based on detection
    """
    if audio_path and Path(audio_path).exists():
        # Custom audio
        autoplay_audio(audio_path)
    else:
        # Default beep (jika tidak ada audio)
        st.warning(f"🔔 Audio warning aktif: {warning_type}")


# ============================================
# INFERENCE PIPELINE
# ============================================
class StreamlitInferencePipeline:
    def __init__(self, model_path, num_frames=8, target_size=112):
        self.num_frames = num_frames
        self.target_size = target_size
        self.device = torch.device('cpu')  # Force CPU untuk compatibility
        
        # Load model
        self.model = MobileDrowsinessModel(num_classes=4, num_frames=num_frames)
        if Path(model_path).exists():
            # ✅ FIX: Load checkpoint yang berisi metadata
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Ekstrak model_state_dict dari checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Format dari training notebook (dengan metadata)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from checkpoint (Fold {checkpoint.get('fold', 'N/A')}, F1: {checkpoint.get('best_f1', 0):.4f})")
            else:
                # Format langsung state_dict (fallback)
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded model from direct state_dict")
            
            self.model.eval()
            self.model_loaded = True
        else:
            self.model_loaded = False
        
        # Frame buffer
        self.frame_buffer = []
        
        # Class names
        self.class_names = ['Focus', 'Talking', 'Yawning', 'Microsleep']
        self.class_colors = {
            'Focus': '🟢',
            'Talking': '🔵',
            'Yawning': '🟡',
            'Microsleep': '🔴'
        }
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (self.target_size, self.target_size))
        
        # Transform
        frame_tensor = self.transform(frame)
        return frame_tensor
    
    def predict(self, frame):
        """Predict from single frame"""
        if not self.model_loaded:
            return None, 0.0, None
        
        # Preprocess
        frame_tensor = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(frame_tensor)
        
        # Keep last N frames
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)
        
        # Pad if needed
        while len(self.frame_buffer) < self.num_frames:
            self.frame_buffer.append(frame_tensor)
        
        # Stack frames
        frames_tensor = torch.stack(self.frame_buffer[-self.num_frames:])
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, num_frames, C, H, W)
        
        # Predict
        with torch.no_grad():
            output = self.model(frames_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, 0)
        
        prediction = self.class_names[predicted_class.item()]
        confidence = confidence.item()
        probabilities = probabilities.cpu().numpy()
        
        return prediction, confidence, probabilities
    
    def reset(self):
        """Reset buffer"""
        self.frame_buffer = []


# ============================================
# VIDEO PROCESSOR FOR WEBRTC
# ============================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pipeline = None
        self.enable_audio = False
        self.yawning_audio = None
        self.microsleep_audio = None
        self.last_warning_time = 0
        self.warning_cooldown = 2  # seconds
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.pipeline is None or not self.pipeline.model_loaded:
            # Draw "Model not loaded" message
            cv2.putText(img, "Model not loaded", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Predict
        prediction, confidence, probabilities = self.pipeline.predict(img)
        
        if prediction is not None:
            # Get color based on prediction
            color_map = {
                'Focus': (0, 255, 0),      # Green
                'Talking': (255, 0, 0),     # Blue
                'Yawning': (0, 255, 255),   # Yellow
                'Microsleep': (0, 0, 255)   # Red
            }
            color = color_map.get(prediction, (255, 255, 255))
            
            # Draw prediction on frame
            text = f"{prediction}: {confidence*100:.1f}%"
            cv2.putText(img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw warning if drowsy
            if prediction in ['Yawning', 'Microsleep']:
                warning_text = "WARNING: DROWSINESS DETECTED!"
                cv2.putText(img, warning_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Audio warning (with cooldown)
                current_time = time.time()
                if self.enable_audio and (current_time - self.last_warning_time > self.warning_cooldown):
                    self.last_warning_time = current_time
            
            # Draw probabilities bar
            y_offset = 100
            for i, (class_name, prob) in enumerate(zip(self.pipeline.class_names, probabilities)):
                bar_length = int(prob * 200)
                bar_color = color_map.get(class_name, (255, 255, 255))
                
                # Draw bar background
                cv2.rectangle(img, (10, y_offset + i*30), (210, y_offset + i*30 + 20), 
                             (50, 50, 50), -1)
                # Draw probability bar
                cv2.rectangle(img, (10, y_offset + i*30), (10 + bar_length, y_offset + i*30 + 20), 
                             bar_color, -1)
                # Draw text
                cv2.putText(img, f"{class_name[:4]}: {prob*100:.0f}%", (10, y_offset + i*30 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================
# STREAMLIT APP
# ============================================
def main():
    st.set_page_config(
        page_title="Deteksi Kantuk Pengemudi",
        page_icon="🚗",
        layout="wide"
    )
    
    # Title
    st.title("🚗 Sistem Deteksi Kantuk Pengemudi")
    st.markdown("**NetraSiaga** - Deteksi real-time kondisi pengemudi dari video wajah")
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Konfigurasi")
    
    # Model path
    model_path = st.sidebar.text_input(
        "Path Model (.pth)",
        value="best_mobile_model.pth",
        help="Path ke file model PyTorch (.pth)"
    )
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = StreamlitInferencePipeline(model_path)
    
    # Check if model loaded
    if not st.session_state.pipeline.model_loaded:
        st.error(f"❌ Model tidak ditemukan di: `{model_path}`")
        st.info("💡 **Cara menggunakan:**\n"
                "1. Training model di notebook\n"
                "2. Save model: `torch.save(model.state_dict(), 'best_mobile_model.pth')`\n"
                "3. Letakkan file `.pth` di folder yang sama dengan `streamlit_app.py`\n"
                "4. Refresh halaman ini")
        return
    
    st.sidebar.success("✅ Model berhasil dimuat!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Model")
    st.sidebar.info(
        f"**Arsitektur:** MobileNetV3 + LSTM\n\n"
        f"**Parameter:** ~4M\n\n"
        f"**Resolusi:** 112x112\n\n"
        f"**Frames:** 8 frames\n\n"
        f"**Kelas:** 4 (Focus, Talking, Yawning, Microsleep)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Panduan")
    st.sidebar.markdown("""
    1. Klik **"START"** untuk memulai webcam
    2. Izinkan akses kamera browser
    3. Posisikan wajah di depan kamera
    4. Prediksi akan berjalan otomatis
    5. Klik **"STOP"** untuk berhenti
    """)
    
    # ============================================
    # REAL-TIME WEBCAM STREAMING
    # ============================================
    st.header("📹 Deteksi Kantuk Real-Time")
    
    st.info("💡 **Pastikan wajah Anda terlihat jelas di kamera untuk hasil terbaik**")
    
    # WebRTC Configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create video processor
    class VideoProcessorFactory:
        def __init__(self):
            self.processor = None
        
        def create(self):
            self.processor = VideoProcessor()
            self.processor.pipeline = st.session_state.pipeline
            return self.processor
    
    if 'video_processor_factory' not in st.session_state:
        st.session_state.video_processor_factory = VideoProcessorFactory()
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=st.session_state.video_processor_factory.create,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Statistics section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 Focus")
        st.markdown("**Kondisi Normal**")
        st.markdown("Pengemudi fokus pada jalan")
    
    with col2:
        st.markdown("### 🟡 Yawning")
        st.markdown("**⚠️ Peringatan**")
        st.markdown("Pengemudi mulai mengantuk")
    
    with col3:
        st.markdown("### 🔴 Microsleep")
        st.markdown("**🚨 BAHAYA!**")
        st.markdown("Segera berhenti dan istirahat!")
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Tips untuk Hasil Terbaik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Lakukan:**
        - Posisikan wajah di tengah frame
        - Pastikan pencahayaan cukup
        - Hindari backlight (cahaya dari belakang)
        - Jaga jarak 30-50 cm dari kamera
        """)
    
    with col2:
        st.markdown("""
        **❌ Hindari:**
        - Gerakan kepala terlalu cepat
        - Pencahayaan terlalu gelap/terang
        - Wajah tertutup masker/topi
        - Background yang terlalu ramai
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>🚗 <b>NetraSiaga</b> - Sistem Deteksi Kantuk Pengemudi Berbasis AI</p>"
        "<p>Dibuat dengan ❤️ menggunakan Streamlit | Model: MobileNetV3 + LSTM</p>"
        "<p><small>Untuk keamanan berkendara yang lebih baik</small></p>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()