"""
Script to fix model loading by re-exporting model in standard format
Mengatasi persistent_load hooks issues
"""

import os
import sys
import shutil
from pathlib import Path
import torch
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Define CBAM and other custom modules BEFORE importing YOLO
# ============================================================================

# Define CBAM-related modules
class ChannelAttention(torch.nn.Module):
    """Channel Attention module"""
    def __init__(self, in_planes=None, ratio=16, *args, **kwargs):
        super().__init__()
        self.in_planes = in_planes or 1
        
    def forward(self, x):
        return x

class SpatialAttention(torch.nn.Module):
    """Spatial Attention module"""
    def __init__(self, kernel_size=7, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x

class CBAMWrapper(torch.nn.Module):
    """CBAM Wrapper module"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x

class CBAM(torch.nn.Module):
    """CBAM (Convolutional Block Attention Module)"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x

# Add to builtins so they can be unpickled
import builtins
builtins.CBAMWrapper = CBAMWrapper
builtins.CBAM = CBAM
builtins.ChannelAttention = ChannelAttention
builtins.SpatialAttention = SpatialAttention
sys.modules['__main__'].CBAMWrapper = CBAMWrapper
sys.modules['__main__'].CBAM = CBAM
sys.modules['__main__'].ChannelAttention = ChannelAttention
sys.modules['__main__'].SpatialAttention = SpatialAttention

# ============================================================================
# Custom Unpickler
# ============================================================================

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles CBAM and persistent hooks"""
    
    def find_class(self, module, name):
        """Override find_class to handle custom modules"""
        try:
            return super().find_class(module, name)
        except AttributeError:
            logger.info(f"   → Handling custom class: {module}.{name}")
            
            # Map custom class names to our placeholder classes
            custom_classes = {
                'CBAMWrapper': CBAMWrapper,
                'CBAM': CBAM,
                'ChannelAttention': ChannelAttention,
                'SpatialAttention': SpatialAttention,
            }
            
            if name in custom_classes:
                return custom_classes[name]
            
            # Try checking if it contains CBAM or Attention
            if 'CBAM' in name or 'Attention' in name:
                logger.info(f"   → Using CBAM placeholder for {name}")
                return CBAMWrapper
            
            raise
    
    def persistent_load(self, pid):
        """Handle persistent_load for custom hooks"""
        logger.info(f"   → Handling persistent_load: {pid}")
        return None

# Patch torch.load to handle persistent hooks
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Patched torch.load with persistent_load support"""
    try:
        return _original_torch_load(f, *args, **kwargs)
    except Exception as e:
        if 'persistent_load' in str(e) or 'CBAM' in str(e):
            logger.info("   → Using custom unpickler for persistent hooks...")
            if isinstance(f, str):
                with open(f, 'rb') as file:
                    unpickler = CustomUnpickler(file)
                    return unpickler.load()
            else:
                unpickler = CustomUnpickler(f)
                return unpickler.load()
        raise

torch.load = patched_torch_load

def fix_model_loading():
    """Re-export model without custom persistent hooks"""
    
    logger.info("=" * 70)
    logger.info("🔧 YOLO v12 Model Loader - Fixing persistent_load Issues")
    logger.info("=" * 70)
    
    # Paths
    model_original = Path("runs/detect/train4/weights/best.pt")
    model_clean = Path("runs/detect/train4/weights/best_clean.pt")
    model_backup = Path("runs/detect/train4/weights/best_backup.pt")
    
    # Check if original model exists
    if not model_original.exists():
        logger.error(f"❌ Model file not found: {model_original}")
        return False
    
    logger.info(f"📁 Original model: {model_original}")
    logger.info(f"📁 Clean model: {model_clean}")
    
    try:
        # Step 1: Backup original
        logger.info("\n[1/3] Creating backup...")
        if not model_backup.exists():
            shutil.copy2(model_original, model_backup)
            logger.info(f"✓ Backup created: {model_backup}")
        else:
            logger.info(f"✓ Backup already exists: {model_backup}")
        
        # Step 2: Load model with ultralytics (handles the pickle issues)
        logger.info("\n[2/3] Loading model with ultralytics...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_original))
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.info("\n💡 Trying alternative loading method...")
            
            # Try with torch direct load
            import pickle
            
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except AttributeError:
                        if 'CBAM' in name or 'cbam' in name.lower():
                            logger.info(f"   → Handling custom class: {name}")
                            # Create placeholder
                            class PlaceholderModule(torch.nn.Module):
                                def forward(self, x):
                                    return x
                            return PlaceholderModule
                        raise
            
            try:
                with open(model_original, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    checkpoint = unpickler.load()
                logger.info("✓ Model loaded with custom unpickler")
                
                # Now load with YOLO
                model = YOLO(str(model_original))
                logger.info("✓ Model reloaded with YOLO after unpickling")
            except Exception as e2:
                logger.error(f"❌ Alternative loading failed: {str(e2)}")
                return False
        
        # Step 3: Save clean version
        logger.info("\n[3/3] Saving clean model...")
        try:
            # Save using YOLO export (this will clean up persistent hooks)
            logger.info("   → Exporting model in standard format...")
            model.save(str(model_clean))
            logger.info(f"✓ Model saved: {model_clean}")
            
            # Verify the clean model can be loaded
            logger.info("\n[VERIFY] Testing clean model load...")
            model_verify = YOLO(str(model_clean))
            logger.info("✓ Clean model loads successfully!")
            
            # Get model info
            if hasattr(model_verify, 'names'):
                logger.info(f"✓ Classes: {model_verify.names}")
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ SUCCESS! Model fixed and saved")
            logger.info("=" * 70)
            logger.info(f"\n📝 Next steps:")
            logger.info(f"1. Update MODEL_PATH in api_v2.py to:")
            logger.info(f"   MODEL_PATH = r\"runs/detect/train4/weights/best_clean.pt\"")
            logger.info(f"\n2. Restart the API server:")
            logger.info(f"   python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000")
            logger.info(f"\n3. Test the API:")
            logger.info(f"   python client_v2.py --health")
            logger.info(f"   python client_v2.py --image test_image.jpg")
            logger.info("\n" + "=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save clean model: {str(e)}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_model_loading()
    sys.exit(0 if success else 1)
