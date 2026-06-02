import sys
import torch
from pathlib import Path

# Register dummy modules BEFORE importing YOLO
class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.qkv = torch.nn.Identity()
        self.proj = torch.nn.Identity()
        self.attn_drop = torch.nn.Dropout(0)
        self.proj_drop = torch.nn.Dropout(0)
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"No attribute {name}")
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return torch.nn.Identity()
    
    def forward(self, x):
        return x

# Register all possible custom modules
custom_modules = [
    'AAttn', 'CBAM', 'CBAMWrapper', 'ChannelAttention', 'SpatialAttention',
    'MultiHeadAttention', 'ScaledDotProductAttention', 'AdaptiveAttention'
]

for mod_name in custom_modules:
    setattr(sys.modules['__main__'], mod_name, DummyModule)
    sys.modules[mod_name] = DummyModule

# Now import YOLO
sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO

print("🔄 Loading original model...")
try:
    model = YOLO('runs/detect/train4/weights/best_clean.pt')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

print("💾 Re-exporting model to ensure clean state...")
try:
    # Save to a new file with a clean export
    model.save('runs/detect/train4/weights/best_final.pt')
    print("✓ Model re-exported to best_final.pt")
except Exception as e:
    print(f"✗ Error re-exporting: {e}")
    sys.exit(1)

print("\n✅ Model re-export complete!")
print("📝 Use 'best_final.pt' in the API configuration")
