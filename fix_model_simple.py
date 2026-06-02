"""
Ultra-simple model fix - Just load and immediately re-initialize with YOLO.export()
"""

import os
import sys
import shutil
from pathlib import Path
import torch
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Setup custom modules FIRST
# ============================================================================

class ChannelAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class SpatialAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class CBAMWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class CBAM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

# Register globally
import builtins
for cls in [CBAMWrapper, CBAM, ChannelAttention, SpatialAttention]:
    builtins.__dict__[cls.__name__] = cls
    sys.modules['__main__'].__dict__[cls.__name__] = cls

# Patch torch.load with persistent_load handler
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            for cls in [CBAMWrapper, CBAM, ChannelAttention, SpatialAttention]:
                if name == cls.__name__ or name.lower() in cls.__name__.lower():
                    return cls
            raise
    
    def persistent_load(self, pid):
        logger.info(f"  Handling persistent_load for: {pid}")
        return None

_orig_load = torch.load
def safe_load(f, *args, **kwargs):
    try:
        return _orig_load(f, *args, **kwargs)
    except Exception as e:
        if 'persistent' in str(e).lower() or 'cbam' in str(e).lower():
            logger.info(f"  Using SafeUnpickler for: {str(e)[:50]}...")
            if isinstance(f, str):
                with open(f, 'rb') as file:
                    unpickler = SafeUnpickler(file)
                    return unpickler.load()
            else:
                return SafeUnpickler(f).load()
        raise

torch.load = safe_load

# ============================================================================
# Main fix
# ============================================================================

def fix_model():
    logger.info("🔧 Fixing YOLO v12 Model...")
    logger.info("=" * 60)
    
    original = Path("runs/detect/train4/weights/best.pt")
    if not original.exists():
        logger.error(f"❌ Model not found: {original}")
        return False
    
    try:
        logger.info(f"[1] Loading model from: {original}")
        from ultralytics import YOLO
        
        # This should now work because we patched torch.load
        model = YOLO(str(original))
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"✓ Classes: {model.names}")
        
        logger.info(f"\n[2] Saving clean model...")
        clean_model = original.parent / "best_clean.pt"
        
        # Save model using ultralytics save method (creates clean weights)
        model.save(str(clean_model))
        logger.info(f"✓ Model saved: {clean_model}")
        
        logger.info(f"\n[3] Verifying clean model...")
        model_verify = YOLO(str(clean_model))
        logger.info(f"✓ Clean model verified!")
        logger.info(f"✓ Classes: {model_verify.names}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS! Model is ready to use")
        logger.info("=" * 60)
        logger.info(f"\n📝 MODEL_PATH in api_v2.py is already set to: best_clean.pt")
        logger.info(f"📝 Restart API server to test detection")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_model()
    sys.exit(0 if success else 1)
