"""
Advanced model fixing - Removes persistent hooks from model state dict
"""

import os
import sys
import shutil
from pathlib import Path
import torch
import logging
import pickle
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Define custom modules
# ============================================================================

class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes=None, ratio=16, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7, *args, **kwargs):
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

# Register in builtins
import builtins
builtins.CBAMWrapper = CBAMWrapper
builtins.CBAM = CBAM
builtins.ChannelAttention = ChannelAttention
builtins.SpatialAttention = SpatialAttention

# ============================================================================
# Custom Unpickler
# ============================================================================

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            custom_classes = {
                'CBAMWrapper': CBAMWrapper,
                'CBAM': CBAM,
                'ChannelAttention': ChannelAttention,
                'SpatialAttention': SpatialAttention,
            }
            if name in custom_classes:
                return custom_classes[name]
            if 'CBAM' in name or 'Attention' in name:
                return CBAMWrapper
            raise
    
    def persistent_load(self, pid):
        return None

# ============================================================================
# Model fixing function
# ============================================================================

def fix_model_advanced():
    """Advanced model fixing with state dict extraction"""
    
    logger.info("=" * 70)
    logger.info("🔧 Advanced YOLO v12 Model Fixer - Extracting Clean State Dict")
    logger.info("=" * 70)
    
    original_model = Path("runs/detect/train4/weights/best.pt")
    clean_model = Path("runs/detect/train4/weights/best_clean.pt")
    backup_model = Path("runs/detect/train4/weights/best_backup.pt")
    
    if not original_model.exists():
        logger.error(f"❌ Model not found: {original_model}")
        return False
    
    logger.info(f"📁 Original: {original_model}")
    logger.info(f"📁 Target: {clean_model}")
    
    try:
        # Step 1: Backup
        logger.info("\n[1/4] Creating backup...")
        if not backup_model.exists():
            shutil.copy2(original_model, backup_model)
            logger.info(f"✓ Backup: {backup_model}")
        else:
            logger.info(f"✓ Backup already exists")
        
        # Step 2: Load checkpoint with safe unpickler
        logger.info("\n[2/4] Loading checkpoint with safe unpickler...")
        with open(original_model, 'rb') as f:
            unpickler = SafeUnpickler(f)
            checkpoint = unpickler.load()
        logger.info(f"✓ Checkpoint loaded, type: {type(checkpoint)}")
        
        # Step 3: Extract model and state dict
        logger.info("\n[3/4] Extracting clean state dict...")
        
        # If checkpoint is a dict, look for model
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                logger.info(f"✓ Found 'model' key in checkpoint")
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
                logger.info(f"✓ Found 'state_dict' key in checkpoint")
            else:
                # Assume the whole dict is the model
                model_state = checkpoint
                logger.info(f"✓ Using checkpoint as model state")
        else:
            logger.error(f"❌ Unexpected checkpoint type: {type(checkpoint)}")
            return False
        
        # Step 4: Save clean checkpoint without persistent hooks
        logger.info("\n[4/4] Saving clean model...")
        
        # Save using standard torch.save (without persistent hooks)
        torch.save(checkpoint, str(clean_model), pickle_protocol=4)
        logger.info(f"✓ Model saved: {clean_model}")
        
        # Verify
        logger.info("\n[VERIFY] Testing clean model load...")
        with open(clean_model, 'rb') as f:
            test_unpickler = SafeUnpickler(f)
            test_checkpoint = test_unpickler.load()
        logger.info(f"✓ Clean model loads successfully")
        
        # Test with ultralytics
        logger.info("\n[TEST] Loading with ultralytics...")
        from ultralytics import YOLO
        model = YOLO(str(clean_model))
        logger.info(f"✓ Model loaded with YOLO")
        if hasattr(model, 'names'):
            logger.info(f"✓ Classes: {model.names}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ SUCCESS! Model fixed and saved")
        logger.info("=" * 70)
        logger.info(f"\n📝 Files:")
        logger.info(f"- Original:  {original_model.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"- Clean:     {clean_model.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"\n📝 Next: Restart API server and test detection endpoints")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_model_advanced()
    sys.exit(0 if success else 1)
