# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import time
from pathlib import Path
import timm
import base64

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
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
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
    
    # ============================================
    # AUDIO WARNING CONFIGURATION
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔊 Konfigurasi Audio Warning")
    
    enable_audio = st.sidebar.checkbox("Aktifkan Audio Warning", value=True)
    
    # Upload custom audio files
    yawning_audio = None
    microsleep_audio = None
    
    if enable_audio:
        st.sidebar.markdown("#### Upload Suara Custom (Opsional)")
        
        yawning_audio_file = st.sidebar.file_uploader(
            "🟡 Audio untuk Yawning",
            type=['mp3', 'wav', 'ogg'],
            help="Format: MP3, WAV, OGG"
        )
        
        microsleep_audio_file = st.sidebar.file_uploader(
            "🔴 Audio untuk Microsleep",
            type=['mp3', 'wav', 'ogg'],
            help="Format: MP3, WAV, OGG"
        )
        
        # Save uploaded audio files
        if yawning_audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp.write(yawning_audio_file.read())
                yawning_audio = tmp.name
                st.sidebar.success("✅ Audio Yawning uploaded!")
        
        if microsleep_audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp.write(microsleep_audio_file.read())
                microsleep_audio = tmp.name
                st.sidebar.success("✅ Audio Microsleep uploaded!")
        
        # Store in session state
        st.session_state.yawning_audio = yawning_audio
        st.session_state.microsleep_audio = microsleep_audio
        st.session_state.enable_audio = enable_audio
    
    # Mode selection
    mode = st.sidebar.radio(
        "Pilih Mode",
        ["📸 Upload Gambar", "🎥 Upload Video", "📹 Webcam (Live)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Model")
    st.sidebar.info(
        f"**Arsitektur:** MobileNetV3 + LSTM\n\n"
        f"**Parameter:** ~4M\n\n"
        f"**Resolusi:** 112x112\n\n"
        f"**Frames:** 8 frames\n\n"
        f"**Kelas:** 4 (Focus, Talking, Yawning, Microsleep)"
    )
    
    # ============================================
    # MODE 1: UPLOAD GAMBAR
    # ============================================
    if mode == "📸 Upload Gambar":
        st.header("📸 Deteksi dari Gambar")
        
        uploaded_file = st.file_uploader(
            "Upload gambar wajah pengemudi",
            type=['jpg', 'jpeg', 'png'],
            help="Format: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)
            
            # Predict
            st.session_state.pipeline.reset()
            prediction, confidence, probabilities = st.session_state.pipeline.predict(img_array)
            
            with col2:
                st.subheader("Hasil Prediksi")
                
                # Display prediction
                icon = st.session_state.pipeline.class_colors[prediction]
                st.markdown(f"## {icon} **{prediction}**")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                
                # Progress bars
                st.markdown("### Probabilitas per Kelas:")
                for i, (class_name, prob) in enumerate(zip(st.session_state.pipeline.class_names, probabilities)):
                    icon = st.session_state.pipeline.class_colors[class_name]
                    st.progress(float(prob), text=f"{icon} {class_name}: {prob*100:.1f}%")
            
            # Alert & Audio Warning
            if prediction == 'Yawning':
                st.error(f"🚨 **PERINGATAN!** Pengemudi terdeteksi {prediction}!")
                if enable_audio:
                    play_warning_sound(st.session_state.get('yawning_audio'), "yawning")
            elif prediction == 'Microsleep':
                st.error(f"🚨 **BAHAYA!** Pengemudi terdeteksi {prediction}!")
                if enable_audio:
                    play_warning_sound(st.session_state.get('microsleep_audio'), "microsleep")
    
    # ============================================
    # MODE 2: UPLOAD VIDEO
    # ============================================
    elif mode == "🎥 Upload Video":
        st.header("🎥 Deteksi dari Video")
        
        uploaded_video = st.file_uploader(
            "Upload video wajah pengemudi",
            type=['mp4', 'avi', 'mov'],
            help="Format: MP4, AVI, MOV"
        )
        
        if uploaded_video is not None:
            # Save uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            # Process video
            st.info("⏳ Memproses video...")
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create placeholders
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            alert_placeholder = st.empty()
            
            # Results storage
            predictions_list = []
            warning_triggered = False
            last_warning_frame = 0
            
            st.session_state.pipeline.reset()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Predict every 5 frames (for speed)
                if frame_count % 5 == 0:
                    prediction, confidence, probabilities = st.session_state.pipeline.predict(frame)
                    predictions_list.append({
                        'frame': frame_count,
                        'prediction': prediction,
                        'confidence': confidence
                    })
                    
                    # Draw on frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    icon = st.session_state.pipeline.class_colors[prediction]
                    text = f"{icon} {prediction} ({confidence*100:.1f}%)"
                    
                    # Display frame
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Frame {frame_count}/{total_frames} - {text}")
                    
                    # Trigger audio warning (cooldown 30 frames)
                    if prediction in ['Yawning', 'Microsleep'] and (frame_count - last_warning_frame > 30):
                        if prediction == 'Yawning':
                            alert_placeholder.error("🚨 **PERINGATAN!** Pengemudi menguap!")
                            if enable_audio:
                                play_warning_sound(st.session_state.get('yawning_audio'), "yawning")
                        else:
                            alert_placeholder.error("🚨 **BAHAYA!** Pengemudi terdeteksi Microsleep!")
                            if enable_audio:
                                play_warning_sound(st.session_state.get('microsleep_audio'), "microsleep")
                        
                        last_warning_frame = frame_count
                        time.sleep(0.5)  # Brief pause for audio
            
            cap.release()
            
            st.success(f"✅ Video selesai diproses! Total {len(predictions_list)} prediksi")
            
            # Summary
            st.subheader("📊 Ringkasan Hasil")
            import pandas as pd
            df = pd.DataFrame(predictions_list)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frame", f"{frame_count}")
            with col2:
                most_common = df['prediction'].mode()[0]
                st.metric("Kondisi Dominan", most_common)
            with col3:
                avg_conf = df['confidence'].mean()
                st.metric("Rata-rata Confidence", f"{avg_conf*100:.1f}%")
            with col4:
                danger_count = len(df[df['prediction'].isin(['Yawning', 'Microsleep'])])
                st.metric("⚠️ Warning Count", danger_count)
            
            # Distribution chart
            st.bar_chart(df['prediction'].value_counts())
    
    # ============================================
    # MODE 3: WEBCAM
    # ============================================
    else:
        st.header("📹 Deteksi Live dari Webcam")
        
        st.warning("⚠️ Fitur webcam memerlukan akses kamera browser Anda")
        
        # Camera input
        camera_image = st.camera_input("Ambil foto dari webcam")
        
        if camera_image is not None:
            # Read image
            image = Image.open(camera_image)
            img_array = np.array(image)
            
            # Predict
            st.session_state.pipeline.reset()
            prediction, confidence, probabilities = st.session_state.pipeline.predict(img_array)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, use_container_width=True)
            
            with col2:
                icon = st.session_state.pipeline.class_colors[prediction]
                st.markdown(f"## {icon} **{prediction}**")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                
                st.markdown("### Probabilitas:")
                for class_name, prob in zip(st.session_state.pipeline.class_names, probabilities):
                    icon = st.session_state.pipeline.class_colors[class_name]
                    st.progress(float(prob), text=f"{icon} {class_name}: {prob*100:.1f}%")
            
            # Alert & Audio Warning
            if prediction == 'Yawning':
                st.error(f"🚨 **PERINGATAN!** Pengemudi terdeteksi {prediction}!")
                if enable_audio:
                    play_warning_sound(st.session_state.get('yawning_audio'), "yawning")
            elif prediction == 'Microsleep':
                st.error(f"🚨 **BAHAYA!** Pengemudi terdeteksi {prediction}!")
                if enable_audio:
                    play_warning_sound(st.session_state.get('microsleep_audio'), "microsleep")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>Dibuat dengan ❤️ menggunakan Streamlit | Model: MobileNetV3 + LSTM</p>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()