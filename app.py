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
    1. Klik **"Start Detection"** 
    2. Posisikan wajah di depan kamera
    3. Prediksi berjalan otomatis setiap 8 frame
    4. Klik **"Stop Detection"** untuk berhenti
    """)
    
    # ============================================
    # REAL-TIME WEBCAM DETECTION (OpenCV Based)
    # ============================================
    st.header("📹 Deteksi Kantuk Real-Time")
    
    st.info("💡 **Pastikan wajah Anda terlihat jelas di kamera untuk hasil terbaik**")
    
    # Frame skip configuration
    frame_skip = st.sidebar.slider(
        "Process Every N Frames",
        min_value=1,
        max_value=10,
        value=3,
        help="Process setiap N frame (lebih besar = lebih cepat tapi kurang smooth)"
    )
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("🎬 Start Detection", type="primary", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    if start_button:
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.session_state.pipeline.reset()
    
    if stop_button:
        st.session_state.running = False
    
    # Video detection loop
    if st.session_state.running:
        # Create placeholders
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Metric placeholders
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_focus = st.empty()
        with col2:
            metric_talking = st.empty()
        with col3:
            metric_yawning = st.empty()
        with col4:
            metric_microsleep = st.empty()
        
        alert_placeholder = st.empty()
        
        # Open webcam with DirectShow backend (Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # If DirectShow fails, try default backend
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
            st.session_state.running = False
        else:
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            st.success("✅ Kamera berhasil dibuka")
            
            frame_counter = 0
            last_prediction = None
            last_confidence = 0
            last_probabilities = None
            
            while st.session_state.running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Gagal membaca frame dari kamera")
                    break
                
                frame_counter += 1
                st.session_state.frame_count = frame_counter
                
                # Process every N frames
                if frame_counter % frame_skip == 0:
                    # Predict
                    prediction, confidence, probabilities = st.session_state.pipeline.predict(frame)
                    
                    if prediction is not None:
                        last_prediction = prediction
                        last_confidence = confidence
                        last_probabilities = probabilities
                
                # Draw on frame
                if last_prediction is not None:
                    # Color mapping
                    color_map = {
                        'Focus': (0, 255, 0),
                        'Talking': (255, 0, 0),
                        'Yawning': (0, 255, 255),
                        'Microsleep': (0, 0, 255)
                    }
                    color = color_map.get(last_prediction, (255, 255, 255))
                    
                    # Draw prediction
                    text = f"{last_prediction}: {last_confidence*100:.1f}%"
                    cv2.putText(frame, text, (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Draw warning
                    if last_prediction in ['Yawning', 'Microsleep']:
                        warning_text = "WARNING: DROWSINESS!"
                        cv2.putText(frame, warning_text, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        # Alert message
                        if last_prediction == 'Yawning':
                            alert_placeholder.warning("⚠️ **PERINGATAN!** Pengemudi mulai mengantuk (Yawning)")
                        else:
                            alert_placeholder.error("🚨 **BAHAYA!** Pengemudi terdeteksi Microsleep!")
                    else:
                        alert_placeholder.empty()
                    
                    # Draw probability bars
                    if last_probabilities is not None:
                        y_offset = 120
                        for i, (class_name, prob) in enumerate(zip(st.session_state.pipeline.class_names, last_probabilities)):
                            bar_length = int(prob * 300)
                            bar_color = color_map.get(class_name, (255, 255, 255))
                            
                            # Background
                            cv2.rectangle(frame, (10, y_offset + i*40), (310, y_offset + i*40 + 30), 
                                         (50, 50, 50), -1)
                            # Bar
                            cv2.rectangle(frame, (10, y_offset + i*40), (10 + bar_length, y_offset + i*40 + 30), 
                                         bar_color, -1)
                            # Text
                            cv2.putText(frame, f"{class_name}: {prob*100:.0f}%", 
                                       (15, y_offset + i*40 + 22), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Update metrics
                        metric_focus.metric("🟢 Focus", f"{last_probabilities[0]*100:.1f}%")
                        metric_talking.metric("🔵 Talking", f"{last_probabilities[1]*100:.1f}%")
                        metric_yawning.metric("🟡 Yawning", f"{last_probabilities[2]*100:.1f}%")
                        metric_microsleep.metric("🔴 Microsleep", f"{last_probabilities[3]*100:.1f}%")
                
                # Frame info
                cv2.putText(frame, f"Frame: {frame_counter}", (frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Status
                status_placeholder.text(f"⚡ Processing... Frame {frame_counter} | Model buffer: {len(st.session_state.pipeline.frame_buffer)}/8")
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            # Release camera
            cap.release()
            st.info("⏹️ Detection stopped")
    
    elif not st.session_state.running and st.session_state.frame_count > 0:
        st.info(f"📊 Session ended. Processed {st.session_state.frame_count} frames")
    
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