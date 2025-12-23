# 👁️ NetraSiaga - Sistem Deteksi Kantuk Pengemudi

Aplikasi deteksi kantuk pengemudi berbasis AI menggunakan MobileNetV3 + LSTM untuk deteksi real-time.

## 🚀 Features

- ✅ Deteksi real-time menggunakan WebRTC (kamera browser)
- ✅ Upload video untuk analisis
- ✅ 4 kelas deteksi: Focus, Talking, Yawning, Microsleep
- ✅ Alert audio untuk kondisi berbahaya
- ✅ Support ONNX dan PyTorch

## 📦 Installation

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy ke Streamlit Cloud

1. **Push ke GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/username/netra-siaga.git
   git push -u origin main
   ```

2. **Deploy di Streamlit Cloud**
   - Buka [share.streamlit.io](https://share.streamlit.io)
   - Login dengan GitHub
   - Klik "New app"
   - Pilih repository: `username/netra-siaga`
   - Main file: `app.py`
   - Klik "Deploy"

3. **Upload Model File**
   - Setelah deploy, upload file model (.pth atau .onnx) ke repository
   - Atau gunakan Git LFS untuk file besar:
     ```bash
     git lfs install
     git lfs track "*.pth"
     git lfs track "*.onnx"
     git add .gitattributes
     git add best_mobile_model.pth
     git commit -m "Add model file"
     git push
     ```

## 🔧 Configuration

### STUN Server

Aplikasi menggunakan Google STUN server untuk WebRTC:
```python
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
```

Anda juga bisa menggunakan STUN server lain atau TURN server untuk koneksi yang lebih stabil:

```python
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:your-turn-server.com:3478"],
            "username": "your-username",
            "credential": "your-password"
        }
    ]
})
```

### Model Files

Aplikasi mendukung 3 tipe model:
- **PyTorch (.pth)**: Model standar
- **PyTorch Quantized (.pth)**: Model yang sudah di-quantize untuk inference lebih cepat
- **ONNX (.onnx)**: Format universal, inference paling cepat

Letakkan file model di root folder project.

## 📁 File Structure

```
netra-siaga/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── packages.txt               # System dependencies
├── .streamlit/
│   └── config.toml           # Streamlit config
├── best_mobile_model.pth     # PyTorch model
├── best_mobile_model.onnx    # ONNX model (optional)
└── warning.mp3               # Alert sound (optional)
```

## 🎯 Usage

### Real-time Detection
1. Buka aplikasi
2. Aktifkan suara peringatan (opsional)
3. Klik tab "Deteksi Real-Time"
4. Klik "START" pada video player
5. Izinkan akses kamera browser
6. Sistem akan mulai mendeteksi

### Video Analysis
1. Klik tab "Upload Video"
2. Upload file video (MP4, AVI, MOV, dll)
3. Klik "Proses Video"
4. Tunggu hingga selesai
5. Lihat hasil analisis

## 🔊 Audio Alert

Aplikasi mendukung alert audio untuk kondisi berbahaya (Yawning/Microsleep):
- Letakkan file `warning.mp3` di root folder
- Aktifkan suara dengan tombol di sidebar
- Audio akan otomatis play saat terdeteksi kantuk

## ⚡ Performance Tips

1. **Gunakan ONNX model** untuk inference paling cepat
2. **Kurangi resolusi video** jika lag
3. **Adjust frame_skip** di sidebar untuk sampling rate
4. **Matikan YOLO** jika tidak perlu crop face (lebih cepat)

## 🐛 Troubleshooting

### Kamera tidak berfungsi
- Pastikan browser memiliki izin akses kamera
- Coba refresh halaman dan izinkan akses lagi
- Gunakan browser modern (Chrome, Firefox, Edge)

### Model tidak ditemukan
- Pastikan file model ada di root folder
- Cek nama file sesuai dengan input di sidebar
- Upload ulang file model ke repository

### Audio tidak bunyi
- Klik tombol "Aktifkan Suara" di sidebar
- Pastikan file `warning.mp3` ada
- Cek volume browser/device

## 📝 License

MIT License - feel free to use for your projects!

## 👨‍💻 Developer

Made with ❤️ for safer driving

---

**Waspada di Jalan, Selamat di Tujuan** 🚗
