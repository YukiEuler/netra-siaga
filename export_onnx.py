import torch
import torch.nn as nn
import timm
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np

class MobileDrowsinessModel(nn.Module):
    """Mobile-Optimized Architecture for Drowsiness Detection"""
    def __init__(self, num_classes=4, num_frames=8, dropout=0.2):
        super().__init__()
        self.num_frames = num_frames
        
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 112, 112)
            dummy_feat = self.backbone(dummy)
        self.feature_dim = dummy_feat.shape[1]
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.lstm_feature_dim = 128
        
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x_flat = x.view(batch_size * num_frames, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(batch_size, num_frames, self.feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_features = h_n[-1]
        output = self.classifier(final_features)
        return output


def export_to_onnx(model_path, output_path):
    """Export PyTorch model to ONNX format"""
    print("Loading PyTorch model...")
    model = MobileDrowsinessModel(num_classes=4, num_frames=8)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 8, 3, 112, 112)
    
    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference
    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # PyTorch inference
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_out = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
    
    print("✓ ONNX export successful! Outputs match PyTorch.")
    print(f"✓ Saved to: {output_path}")
    
    # Benchmark
    import time
    
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        pytorch_time = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start) / 100
    
    print(f"\n{'='*50}")
    print(f"PyTorch:  {pytorch_time*1000:.2f} ms/inference")
    print(f"ONNX:     {onnx_time*1000:.2f} ms/inference")
    print(f"Speedup:  {pytorch_time/onnx_time:.2f}x")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    model_path = "best_mobile_model.pth"
    output_path = "best_mobile_model.onnx"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
    else:
        export_to_onnx(model_path, output_path)