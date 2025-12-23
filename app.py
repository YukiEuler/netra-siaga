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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# STUN server configuration untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Optional YOLO face detector
# Uses ultralytics YOLO to crop the driver's face before classification
class YOLOFaceDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3, iou=0.5):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Add it to requirements and reinstall.") from exc
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def extract_face(self, frame):
        """Run YOLO and return the union crop of all detected faces (BGR)."""
        if frame is None:
            return None

        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False, device="cpu")
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        x1 = np.min(boxes[:, 0])
        y1 = np.min(boxes[:, 1])
        x2 = np.max(boxes[:, 2])
        y2 = np.max(boxes[:, 3])

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

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
    def __init__(self, model_path, num_frames=8, target_size=112, face_detector=None, frame_skip=3, use_onnx=False):
        self.num_frames = num_frames
        self.target_size = target_size
        self.device = torch.device('cpu')  # Force CPU untuk compatibility
        self.face_detector = face_detector
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        
        # Load model
        if self.use_onnx and model_path.endswith('.onnx'):
            if Path(model_path).exists():
                print("Loading ONNX model...")
                self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.model = None
                self.model_loaded = True
                print("✓ ONNX model loaded successfully")
            else:
                self.model_loaded = False
                self.ort_session = None
        else:
            self.model = MobileDrowsinessModel(num_classes=4, num_frames=num_frames)
            self.ort_session = None
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if checkpoint.get('quantized', False):
                        print(f"✓ Loaded quantized model from checkpoint")
                    else:
                        print(f"✓ Loaded model from checkpoint (Fold {checkpoint.get('fold', 'N/A')}, F1: {checkpoint.get('best_f1', 0):.4f})")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("✓ Loaded model from direct state_dict")
                
                self.model.eval()
                self.model_loaded = True
            else:
                self.model_loaded = False
        
        self.frame_buffer = []
        self.last_bbox = None  # Cache last detected bbox
        self.yolo_frame_counter = 0  # Counter for YOLO skip
        
        self.class_names = ['Focus', 'Talking', 'Yawning', 'Microsleep']
        self.class_colors = {
            'Focus': '🟢',
            'Talking': '🔵',
            'Yawning': '🟡',
            'Microsleep': '🔴'
        }
        
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
    
    def predict(self, frame, force_add=False):
        """Predict from single frame"""
        import time
        start_total = time.time()
        
        if not self.model_loaded:
            return None, 0.0, None

        # Increment frame counter
        self.frame_counter += 1
        
        # Store raw frame for union bbox calculation
        start_copy = time.time()
        raw_frame = frame.copy()
        print(f"[TIMING] Frame copy: {(time.time() - start_copy)*1000:.2f}ms")
        
        # Only add frame to raw buffer every N frames (frame_skip)
        if force_add or self.frame_counter % self.frame_skip == 0:
            # YOLO face detection - only run every 5 frames to reduce overhead
            if self.face_detector is not None:
                self.yolo_frame_counter += 1
                # Only run YOLO every 5 buffer updates (15 actual frames if frame_skip=3)
                if self.yolo_frame_counter % 5 == 0 or self.last_bbox is None:
                    start_yolo = time.time()
                    # Detect face only on current frame
                    results = self.face_detector.model.predict(
                        raw_frame, 
                        conf=self.face_detector.conf, 
                        iou=self.face_detector.iou, 
                        verbose=False, 
                        device="cpu"
                    )
                    
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        x1 = np.min(boxes[:, 0])
                        y1 = np.min(boxes[:, 1])
                        x2 = np.max(boxes[:, 2])
                        y2 = np.max(boxes[:, 3])
                        
                        h, w = raw_frame.shape[:2]
                        x1 = max(0, min(w - 1, int(x1)))
                        x2 = max(0, min(w, int(x2)))
                        y1 = max(0, min(h - 1, int(y1)))
                        y2 = max(0, min(h, int(y2)))
                        
                        if x2 > x1 and y2 > y1:
                            self.last_bbox = (x1, y1, x2, y2)
                    
                    print(f"[TIMING] YOLO detection (1 frame): {(time.time() - start_yolo)*1000:.2f}ms")
                else:
                    print(f"[TIMING] YOLO skipped (using cached bbox)")
                
                # Preprocess with detected bbox
                start_preprocess = time.time()
                if self.last_bbox is not None:
                    x1, y1, x2, y2 = self.last_bbox
                    cropped = raw_frame[y1:y2, x1:x2]
                    frame_tensor = self.preprocess_frame(cropped)
                else:
                    frame_tensor = self.preprocess_frame(raw_frame)
                print(f"[TIMING] Preprocessing 1 frame: {(time.time() - start_preprocess)*1000:.2f}ms")
            else:
                # No YOLO, just preprocess
                start_preprocess = time.time()
                frame_tensor = self.preprocess_frame(raw_frame)
                print(f"[TIMING] Preprocessing 1 frame: {(time.time() - start_preprocess)*1000:.2f}ms")
            
            self.frame_buffer.append(frame_tensor)
            if len(self.frame_buffer) > self.num_frames:
                self.frame_buffer.pop(0)
        
        # Fill buffer with current frame if not enough frames yet
        start_fill = time.time()
        fill_count = 0
        while len(self.frame_buffer) < self.num_frames:
            # Use cached bbox instead of re-running YOLO
            if self.face_detector is not None and self.last_bbox is not None:
                x1, y1, x2, y2 = self.last_bbox
                cropped = raw_frame[y1:y2, x1:x2]
                frame_tensor = self.preprocess_frame(cropped)
            else:
                frame_tensor = self.preprocess_frame(raw_frame)
            self.frame_buffer.append(frame_tensor)
            fill_count += 1
        if fill_count > 0:
            print(f"[TIMING] Buffer filling ({fill_count} frames): {(time.time() - start_fill)*1000:.2f}ms")
        
        start_stack = time.time()
        frames_tensor = torch.stack(self.frame_buffer[-self.num_frames:])
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, num_frames, C, H, W)
        print(f"[TIMING] Stack tensors: {(time.time() - start_stack)*1000:.2f}ms")
        
        # ONNX or PyTorch inference
        start_inference = time.time()
        if self.use_onnx and self.ort_session is not None:
            input_data = frames_tensor.numpy()
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
            output = self.ort_session.run(None, ort_inputs)[0]
            output = torch.from_numpy(output)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, 0)
            print(f"[TIMING] ONNX inference: {(time.time() - start_inference)*1000:.2f}ms")
        else:
            with torch.no_grad():
                output = self.model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
            print(f"[TIMING] PyTorch inference: {(time.time() - start_inference)*1000:.2f}ms")
        
        start_post = time.time()
        prediction = self.class_names[predicted_class.item()]
        confidence = confidence.item()
        probabilities = probabilities.cpu().numpy()
        print(f"[TIMING] Post-processing: {(time.time() - start_post)*1000:.2f}ms")
        
        print(f"[TIMING] ===== TOTAL predict(): {(time.time() - start_total)*1000:.2f}ms =====\n")
        
        return prediction, confidence, probabilities
    
    def _get_union_bbox(self, frames):
        """Calculate union bounding box from all frames"""
        all_boxes = []
        
        for frame in frames:
            results = self.face_detector.model.predict(
                frame, 
                conf=self.face_detector.conf, 
                iou=self.face_detector.iou, 
                verbose=False, 
                device="cpu"
            )
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                all_boxes.extend(boxes)
        
        if len(all_boxes) == 0:
            return None
        
        # Calculate union of all boxes
        all_boxes = np.array(all_boxes)
        x1 = np.min(all_boxes[:, 0])
        y1 = np.min(all_boxes[:, 1])
        x2 = np.max(all_boxes[:, 2])
        y2 = np.max(all_boxes[:, 3])
        
        # Get frame dimensions from first frame
        h, w = frames[0].shape[:2]
        
        # Clamp to frame boundaries
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return (x1, y1, x2, y2)
    
    def reset(self):
        """Reset buffer"""
        self.frame_buffer = []
        if hasattr(self, 'raw_frame_buffer'):
            self.raw_frame_buffer = []
        self.frame_counter = 0
        self.last_bbox = None


# ============================================
# WEBRTC VIDEO PROCESSOR
# ============================================
class VideoProcessor:
    """Video processor untuk streamlit-webrtc"""
    def __init__(self, pipeline, alert_audio_bytes=None):
        self.pipeline = pipeline
        self.alert_audio_bytes = alert_audio_bytes
        self.frame_counter = 0
        self.last_prediction = None
        self.last_confidence = 0
        self.last_probabilities = None
        self.lock = threading.Lock()
        
    def recv(self, frame):
        """Process each frame from WebRTC"""
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_counter += 1
        
        # Predict
        with self.lock:
            prediction, confidence, probabilities = self.pipeline.predict(img)
            
            if prediction is not None:
                self.last_prediction = prediction
                self.last_confidence = confidence
                self.last_probabilities = probabilities
        
        # Draw on frame
        if self.last_prediction is not None:
            # Color mapping
            color_map = {
                'Focus': (0, 255, 0),
                'Talking': (255, 0, 0),
                'Yawning': (0, 255, 255),
                'Microsleep': (0, 0, 255)
            }
            color = color_map.get(self.last_prediction, (255, 255, 255))
            
            # Draw prediction text
            text = f"{self.last_prediction}: {self.last_confidence*100:.1f}%"
            cv2.putText(img, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Draw warning
            if self.last_prediction in ['Yawning', 'Microsleep']:
                warning_text = "WARNING: DROWSINESS!"
                cv2.putText(img, warning_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Draw probability bars
            if self.last_probabilities is not None:
                y_offset = 120
                for i, (class_name, prob) in enumerate(zip(self.pipeline.class_names, self.last_probabilities)):
                    bar_length = int(prob * 300)
                    bar_color = color_map.get(class_name, (255, 255, 255))
                    
                    # Background
                    cv2.rectangle(img, (10, y_offset + i*40), (310, y_offset + i*40 + 30), 
                                 (50, 50, 50), -1)
                    # Bar
                    cv2.rectangle(img, (10, y_offset + i*40), (10 + bar_length, y_offset + i*40 + 30), 
                                 bar_color, -1)
                    # Text
                    cv2.putText(img, f"{class_name}: {prob*100:.0f}%", 
                               (15, y_offset + i*40 + 22), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(img, f"Frame: {self.frame_counter} | Buffer: {len(self.pipeline.frame_buffer)}/8", 
                   (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================
# STREAMLIT APP
# ============================================
def play_audio_autoplay(audio_bytes):
    """Helper function to play audio with Web Audio API for better control"""
    if audio_bytes and st.session_state.get('audio_enabled', False):
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <script>
            // Remove old audio if exists
            if (window.warningAudio) {{
                try {{
                    window.warningAudio.pause();
                    window.warningAudio.currentTime = 0;
                    window.warningAudio = null;
                }} catch(e) {{
                    console.log('Error stopping old audio:', e);
                }}
            }}
            
            // Create new audio with Web Audio API
            window.warningAudio = new Audio('data:audio/mp3;base64,{audio_base64}');
            window.warningAudio.loop = true;
            window.warningAudio.volume = 1.0;
            window.warningAudio.id = 'warning_audio';
            
            // Play audio immediately (user already enabled it)
            window.warningAudio.play().then(function() {{
                console.log('✓ Warning audio playing');
                window.audioPlaying = true;
            }}).catch(function(error) {{
                console.error('✗ Audio play failed:', error);
                window.audioPlaying = false;
                alert('⚠️ Audio tidak bisa diputar! Pastikan Anda sudah mengaktifkan suara.');
            }});
        </script>
        """
        st.markdown(audio_html, unsafe_allow_html=True)


def stop_audio():
    """Stop any playing audio using Web Audio API"""
    stop_html = """
    <script>
        if (window.warningAudio) {
            try {
                window.warningAudio.pause();
                window.warningAudio.currentTime = 0;
                window.warningAudio = null;
                window.audioPlaying = false;
                console.log('✓ Warning audio stopped');
            } catch(e) {
                console.log('✗ Error stopping audio:', e);
            }
        }
    </script>
    """
    st.markdown(stop_html, unsafe_allow_html=True)

def enable_audio():
    """Enable audio by creating silent audio element and getting user permission"""
    enable_html = """
    <script>
        // Create a silent audio to get permission
        var silentAudio = new Audio('data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADhAC7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7v////////////////////////////////////////////////////////////////');
        silentAudio.volume = 0.01;
        silentAudio.play().then(function() {
            console.log('✓ Audio enabled! Browser allows audio playback.');
            window.audioEnabled = true;
        }).catch(function(error) {
            console.error('✗ Failed to enable audio:', error);
            window.audioEnabled = false;
        });
    </script>
    """
    st.markdown(enable_html, unsafe_allow_html=True)

def main():
    frame_skip = 1
    st.set_page_config(
        page_title="NetraSiaga - Deteksi Kantuk Pengemudi",
        page_icon="👁️",
        layout="wide"
    )

    st.markdown("""
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <style>
            /* Menyembunyikan elemen bawaan Streamlit agar lebih clean */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Logo dan Branding */
            .logo-container {
                text-align: center;
                padding: 2rem 0;
                background: white;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: 2px solid #e0e0e0;
            }
            .logo-title {
                font-size: 4rem;
                font-weight: 900;
                color: #1a1a1a;
                text-shadow: none;
                margin: 0;
                letter-spacing: 2px;
            }
            .logo-subtitle {
                font-size: 1.3rem;
                color: #555;
                margin-top: 0.5rem;
                font-weight: 300;
                letter-spacing: 1px;
            }
            .slogan {
                font-size: 1.1rem;
                color: #667eea;
                margin-top: 1rem;
                font-style: italic;
                font-weight: 500;
            }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Logo dan Branding
    st.markdown("""
        <div class="logo-container">
            <div class="logo-title">👁️ NetraSiaga</div>
            <div class="logo-subtitle">Sistem Deteksi Kantuk Pengemudi Berbasis AI</div>
            <div class="slogan">"Mata Cerdas, Penjaga Nyawa"</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Konfigurasi")
    
    # Model type selector
    model_type = st.sidebar.selectbox(
        "Tipe Model",
        ["ONNX (.onnx)", "PyTorch (.pth)", "PyTorch Quantized (.pth)"],
        help="Pilih tipe model untuk inferensi. ONNX lebih cepat!"
    )
    
    # Model path based on type
    if model_type == "ONNX (.onnx)":
        if not ONNX_AVAILABLE:
            st.sidebar.error("❌ ONNX Runtime tidak terinstall. Jalankan: pip install onnxruntime")
            model_path = st.sidebar.text_input(
                "Path Model (.pth)",
                value="best_mobile_model.pth",
                help="Path ke file model PyTorch (.pth)"
            )
            use_onnx = False
        else:
            model_path = st.sidebar.text_input(
                "Path Model (.onnx)",
                value="best_mobile_model.onnx",
                help="Path ke file model ONNX (.onnx)"
            )
            use_onnx = True
    elif model_type == "PyTorch Quantized (.pth)":
        model_path = st.sidebar.text_input(
            "Path Model (.pth)",
            value="best_mobile_model_quantized.pth",
            help="Path ke file model PyTorch quantized (.pth)"
        )
        use_onnx = False
    else:
        model_path = st.sidebar.text_input(
            "Path Model (.pth)",
            value="best_mobile_model.pth",
            help="Path ke file model PyTorch (.pth)"
        )
        use_onnx = False

    # Face detection mode (YOLO optional)
    detection_mode = st.sidebar.selectbox(
        "Deteksi Wajah",
        ["Tanpa YOLO (full frame)", "YOLO Face (crop)"],
        index=0,
        help="YOLO lebih akurat tapi lambat. Tanpa YOLO lebih cepat untuk real-time."
    )

    face_detector = None
    if detection_mode == "YOLO Face (crop)":
        st.sidebar.warning("⚠️ YOLO memperlambat deteksi real-time. Gunakan untuk akurasi maksimal.")
        yolo_model_path = st.sidebar.text_input("YOLO model path", value="yolov8n.pt")
        yolo_conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.3, 0.05)

        needs_new_yolo = (
            'yolo_detector' not in st.session_state
            or st.session_state.get('yolo_model_path') != yolo_model_path
            or st.session_state.get('yolo_conf') != yolo_conf
        )

        if needs_new_yolo:
            try:
                st.session_state.yolo_detector = YOLOFaceDetector(yolo_model_path, conf=yolo_conf)
                st.session_state.yolo_model_path = yolo_model_path
                st.session_state.yolo_conf = yolo_conf
                st.sidebar.success("✅ YOLO face detector aktif")
            except Exception as e:
                st.session_state.yolo_detector = None
                st.sidebar.error(f"Gagal memuat YOLO: {e}")
        face_detector = st.session_state.get('yolo_detector')
        if face_detector is None:
            st.sidebar.warning("YOLO belum aktif, fallback ke full frame.")
    else:
        st.session_state.yolo_detector = None

    # Warning sound configuration
    warning_sound_path = st.sidebar.text_input("Warning sound (mp3)", value="warning.mp3")
    need_reload_sound = (
        'alert_audio_path' not in st.session_state
        or st.session_state.get('alert_audio_path') != warning_sound_path
    )
    if need_reload_sound:
        try:
            st.session_state.alert_audio_bytes = Path(warning_sound_path).read_bytes()
            st.session_state.alert_audio_path = warning_sound_path
            st.sidebar.success("✅ Warning sound siap diputar")
        except Exception as exc:
            st.session_state.alert_audio_bytes = None
            st.session_state.alert_audio_path = warning_sound_path
            st.sidebar.warning(f"Warning sound tidak ditemukan: {exc}")
    
    # Initialize pipeline
    need_new_pipeline = (
        'pipeline' not in st.session_state
        or st.session_state.pipeline.model_path != model_path
        or st.session_state.pipeline.face_detector is not face_detector
        or st.session_state.get('frame_skip') != frame_skip
        or st.session_state.get('use_onnx') != use_onnx
    )
    if need_new_pipeline:
        st.session_state.pipeline = StreamlitInferencePipeline(
            model_path, 
            face_detector=face_detector, 
            frame_skip=3,
            use_onnx=use_onnx
        )
        st.session_state.frame_skip = 3
        st.session_state.use_onnx = use_onnx
    
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
    
    # Audio control section
    st.sidebar.markdown("### 🔊 Kontrol Suara")
    
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = False
    
    if not st.session_state.audio_enabled:
        if st.sidebar.button("🔇 Aktifkan Suara Peringatan", type="primary", use_container_width=True):
            st.session_state.audio_enabled = True
            enable_audio()
            st.sidebar.success("✅ Suara peringatan AKTIF")
            st.rerun()
        st.sidebar.warning("⚠️ Klik untuk mengaktifkan suara warning!")
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.success("🔊 AKTIF")
        with col2:
            if st.button("🔇 Matikan", use_container_width=True):
                st.session_state.audio_enabled = False
                stop_audio()
                st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Model")
    
    # Show ONNX availability
    if not ONNX_AVAILABLE:
        st.sidebar.error("❌ ONNX Runtime tidak tersedia\n\nInstall: `pip install onnxruntime`")
    
    runtime_info = "ONNX Runtime" if use_onnx else "PyTorch"
    if model_type == "PyTorch Quantized (.pth)":
        runtime_info += " (Quantized)"
    
    st.sidebar.info(
        f"**Runtime:** {runtime_info}\n\n"
        f"**Arsitektur:** MobileNetV3 + LSTM\n\n"
        f"**Parameter:** ~4M\n\n"
        f"**Resolusi:** 112x112\n\n"
        f"**Frames:** 8 frames\n\n"
        f"**Kelas:** 4 (Focus, Talking, Yawning, Microsleep)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Panduan")
    st.sidebar.markdown("""
    1. Upload file video (mp4, avi, mov, dll)
    2. Klik **"Proses Video"**
    3. Tunggu hingga selesai
    4. Lihat hasil deteksi
    
    **Tips:**
    - Video dengan resolusi tinggi akan diproses lebih lama
    - Pastikan wajah terlihat jelas di video
    - Durasi video maksimal ~5 menit untuk performa terbaik
    """)
    
    # ============================================
    # DETECTION MODE SELECTION
    # ============================================
    st.header("🎯 Pilih Mode Deteksi")
    
    detection_tab = st.tabs(["📷 Deteksi Real-Time (Kamera)", "📹 Upload Video"])
    
    # Frame skip configuration
    frame_skip = st.sidebar.slider(
        "Frame Sampling Rate",
        min_value=1,
        max_value=10,
        value=1,
        help="Ambil 1 frame setiap N frame untuk buffer (lebih besar = jarak frame lebih jauh)"
    )
    
    # ============================================
    # TAB 1: REAL-TIME DETECTION
    # ============================================
    with detection_tab[0]:
        st.info("💡 **Deteksi real-time menggunakan webcam browser Anda. Model menggunakan 8 frames berurutan untuk prediksi akurat.**")
        
        st.markdown("### 📹 Live Camera Feed")
        
        # Capture references before factory function to avoid session_state access in worker thread
        pipeline = st.session_state.pipeline
        alert_audio = st.session_state.get('alert_audio_bytes')
        
        # Create video processor factory function
        def video_processor_factory():
            return VideoProcessor(pipeline, alert_audio)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=video_processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display metrics in real-time
        if webrtc_ctx.state.playing:
            st.success("✅ Kamera aktif - Deteksi berjalan!")
            
            # Get the current video processor instance
            video_processor = webrtc_ctx.video_processor
            
            # Metric placeholders
            col1, col2, col3, col4 = st.columns(4)
            
            metric_placeholders = {
                'focus': col1.empty(),
                'talking': col2.empty(),
                'yawning': col3.empty(),
                'microsleep': col4.empty()
            }
            
            alert_placeholder = st.empty()
            
            # Update metrics periodically
            last_dangerous_state = False
            audio_playing = False
            
            while webrtc_ctx.state.playing:
                if video_processor and video_processor.last_probabilities is not None:
                    probabilities = video_processor.last_probabilities
                    
                    # Update metrics
                    metric_placeholders['focus'].metric("🟢 Focus", f"{probabilities[0]*100:.1f}%")
                    metric_placeholders['talking'].metric("🔵 Talking", f"{probabilities[1]*100:.1f}%")
                    metric_placeholders['yawning'].metric("🟡 Yawning", f"{probabilities[2]*100:.1f}%")
                    metric_placeholders['microsleep'].metric("🔴 Microsleep", f"{probabilities[3]*100:.1f}%")
                    
                    # Check for dangerous state
                    prediction = video_processor.last_prediction
                    is_dangerous = prediction in ['Yawning', 'Microsleep']
                    
                    if is_dangerous:
                        if prediction == 'Yawning':
                            alert_placeholder.warning("⚠️ **PERINGATAN!** Pengemudi mulai mengantuk (Yawning)")
                        else:
                            alert_placeholder.error("🚨 **BAHAYA!** Pengemudi terdeteksi Microsleep!")
                        
                        # Play audio if enabled
                        if not audio_playing and st.session_state.get('audio_enabled', False):
                            if st.session_state.get('alert_audio_bytes'):
                                play_audio_autoplay(st.session_state.alert_audio_bytes)
                                audio_playing = True
                        
                        last_dangerous_state = True
                    else:
                        # Changed from dangerous to safe
                        if last_dangerous_state:
                            alert_placeholder.empty()
                            if audio_playing:
                                stop_audio()
                                audio_playing = False
                            last_dangerous_state = False
                
                time.sleep(0.1)  # Update every 100ms
        else:
            st.info("👆 Klik 'START' pada video player untuk memulai deteksi menggunakan kamera browser Anda")
            st.markdown("""
            **Catatan:**
            - Browser akan meminta izin akses kamera
            - Pastikan Anda mengizinkan akses kamera untuk aplikasi ini
            - Jika video tidak muncul, coba refresh halaman dan izinkan akses kamera lagi
            """)
    
    # ============================================
    # TAB 2: VIDEO UPLOAD
    # ============================================
    with detection_tab[1]:
        st.info("💡 **Model menggunakan 8 frames berurutan untuk deteksi. Upload video wajah pengemudi!**")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Format yang didukung: MP4, AVI, MOV, MKV, WMV"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            # Show video info
            st.success(f"✅ Video berhasil diupload: {uploaded_file.name}")
            
            # Process button
            process_button = st.button("🎬 Proses Video", type="primary", use_container_width=True)
            
            if process_button:
                # Reset pipeline
                st.session_state.pipeline.reset()
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("❌ Gagal membuka video. Pastikan format video didukung.")
                else:
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    
                    st.info(f"📊 **Info Video:** {total_frames} frames | {fps:.1f} FPS | Durasi: {duration:.1f}s")
                    
                    # Create placeholders
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
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
                    
                    # Statistics
                    frame_counter = 0
                    last_prediction = None
                    last_confidence = 0
                    last_probabilities = None
                    
                    # Counters for statistics
                    prediction_counts = {'Focus': 0, 'Talking': 0, 'Yawning': 0, 'Microsleep': 0}
                    danger_frames = []
                    
                    # Process video
                    while True:
                        ret, frame = cap.read()
                        
                        if not ret:
                            break
                        
                        frame_counter += 1
                        
                        # Update progress
                        progress = frame_counter / total_frames
                        progress_bar.progress(progress)
                        
                        # Predict every frame (frame_skip is handled inside predict())
                        prediction, confidence, probabilities = st.session_state.pipeline.predict(frame)
                        
                        if prediction is not None:
                            last_prediction = prediction
                            last_confidence = confidence
                            last_probabilities = probabilities
                            
                            # Update statistics only when buffer has new frame
                            if frame_counter % frame_skip == 0:
                                prediction_counts[prediction] += 1
                                
                                # Track danger moments
                                if prediction in ['Yawning', 'Microsleep']:
                                    timestamp = frame_counter / fps if fps > 0 else 0
                                    danger_frames.append({
                                        'frame': frame_counter,
                                        'time': timestamp,
                                        'prediction': prediction,
                                        'confidence': confidence
                                    })
                        
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
                        current_time = frame_counter / fps if fps > 0 else 0
                        cv2.putText(frame, f"Frame: {frame_counter}/{total_frames} | Time: {current_time:.1f}s", 
                                   (10, frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Convert to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame (every 5 frames to reduce lag)
                        if frame_counter % 5 == 0:
                            video_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Status
                        status_placeholder.text(f"⚡ Processing frame {frame_counter}/{total_frames} | Buffer: {len(st.session_state.pipeline.frame_buffer)}/8")
                    
                    # Release video
                    cap.release()
                    
                    # Final results
                    st.success("✅ Video selesai diproses!")
                    
                    # Display statistics
                    st.markdown("---")
                    st.markdown("## 📊 Hasil Analisis Video")
                    
                    # Overall statistics
                    col1, col2, col3, col4 = st.columns(4)
                    total_predictions = sum(prediction_counts.values())
                    
                    with col1:
                        focus_pct = (prediction_counts['Focus'] / total_predictions * 100) if total_predictions > 0 else 0
                        st.metric("🟢 Focus", f"{focus_pct:.1f}%", f"{prediction_counts['Focus']} deteksi")
                    
                    with col2:
                        talking_pct = (prediction_counts['Talking'] / total_predictions * 100) if total_predictions > 0 else 0
                        st.metric("🔵 Talking", f"{talking_pct:.1f}%", f"{prediction_counts['Talking']} deteksi")
                    
                    with col3:
                        yawning_pct = (prediction_counts['Yawning'] / total_predictions * 100) if total_predictions > 0 else 0
                        st.metric("🟡 Yawning", f"{yawning_pct:.1f}%", f"{prediction_counts['Yawning']} deteksi")
                    
                    with col4:
                        microsleep_pct = (prediction_counts['Microsleep'] / total_predictions * 100) if total_predictions > 0 else 0
                        st.metric("🔴 Microsleep", f"{microsleep_pct:.1f}%", f"{prediction_counts['Microsleep']} deteksi")
                    
                    # Danger moments
                    if danger_frames:
                        st.markdown("### ⚠️ Momen Berbahaya Terdeteksi")
                        st.warning(f"Ditemukan **{len(danger_frames)} momen berbahaya** dalam video!")
                        
                        # Show table
                        danger_data = []
                        for item in danger_frames[:20]:  # Show first 20
                            mins = int(item['time'] // 60)
                            secs = int(item['time'] % 60)
                            danger_data.append({
                                'Frame': item['frame'],
                                'Waktu': f"{mins:02d}:{secs:02d}",
                                'Kondisi': f"{item['prediction']}",
                                'Confidence': f"{item['confidence']*100:.1f}%"
                            })
                        
                        st.dataframe(danger_data, use_container_width=True)
                        
                        if len(danger_frames) > 20:
                            st.info(f"Menampilkan 20 dari {len(danger_frames)} momen berbahaya")
                    else:
                        st.success("✅ Tidak ada momen berbahaya terdeteksi dalam video!")
                    
                    # Overall assessment
                    danger_pct = ((prediction_counts['Yawning'] + prediction_counts['Microsleep']) / total_predictions * 100) if total_predictions > 0 else 0
                    
                    st.markdown("### 🎯 Penilaian Keseluruhan")
                    if danger_pct > 30:
                        st.error(f"🚨 **SANGAT BERBAHAYA** - {danger_pct:.1f}% video menunjukkan tanda kantuk! Pengemudi tidak aman untuk berkendara.")
                    elif danger_pct > 15:
                        st.warning(f"⚠️ **PERINGATAN** - {danger_pct:.1f}% video menunjukkan tanda kantuk. Pengemudi perlu istirahat.")
                    elif danger_pct > 5:
                        st.info(f"💡 **PERHATIAN** - {danger_pct:.1f}% video menunjukkan tanda kantuk ringan. Sebaiknya waspada.")
                    else:
                        st.success(f"✅ **AMAN** - {danger_pct:.1f}% video menunjukkan tanda kantuk. Pengemudi dalam kondisi baik.")
            
            # Clean up temp file
            try:
                import os
                os.unlink(video_path)
            except:
                pass
        else:
            st.info("👆 Upload video untuk memulai deteksi kantuk")
    
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
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>👁️ NetraSiaga</h3>
            <p style='font-size: 1.1rem; color: #666;'>"Waspada di Jalan, Selamat di Tujuan"</p>
            <p style='margin-top: 1rem;'>Sistem Deteksi Kantuk Pengemudi Berbasis AI</p>
            <p style='font-size: 0.9rem; color: #888;'>Dibuat dengan ❤️ menggunakan Streamlit | Model: MobileNetV3 + LSTM</p>
            <p style='font-size: 0.85rem; color: #999; margin-top: 0.5rem;'>Untuk keamanan berkendara yang lebih baik</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()