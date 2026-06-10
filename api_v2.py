

import os
import io
import base64
import logging
import math
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

import pickle
import sys

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CUSTOM MODULES FOR MODEL LOADING - NOTEBOOK ATTENTION IMPLEMENTATIONS
# ============================================================================

class ChannelAttention(torch.nn.Module):
    """Channel Attention module, matching the training notebook."""
    def __init__(self, channels: int = 1, reduction: int = 16, *args, **kwargs):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(torch.nn.Module):
    """Spatial Attention module, matching the training notebook."""
    def __init__(self, kernel_size: int = 7, *args, **kwargs):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(torch.nn.Module):
    """CBAM (Convolutional Block Attention Module), matching the notebook."""
    def __init__(self, channels: int = 1, reduction: int = 16, *args, **kwargs):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CBAMWrapper(torch.nn.Module):
    """Wrapper used during training: base module followed by real CBAM."""
    def __init__(self, base_module=None, reduction=16, *args, **kwargs):
        super().__init__()
        self.base = base_module
        self.reduction = reduction
        self.cbam = None
        self._initialized = False

        for attr in ("i", "f", "type", "np"):
            if self.base is not None and hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))

    def forward(self, x):
        if self.base is None:
            raise RuntimeError("CBAMWrapper membutuhkan base module asli dari checkpoint")

        x = self.base(x)
        if not torch.is_tensor(x):
            return x

        cbam = self._modules.get("cbam") or self._modules.get("cbam_layer")
        if cbam is not None:
            return cbam(x)

        channels = x.shape[1]
        self.cbam = CBAM(channels, self.reduction).to(x.device)
        self.add_module("cbam_layer", self.cbam)
        self._initialized = True
        return self.cbam(x)


def get_eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """Match the ECA kernel sizing logic used in the training notebook."""
    t = int(abs((math.log2(channels) + b) / gamma))
    return max(t if t % 2 else t + 1, 3)


class ECA(torch.nn.Module):
    """Efficient Channel Attention module, matching the training notebook."""
    def __init__(self, channels: int = 1, gamma: int = 2, b: int = 1, *args, **kwargs):
        super().__init__()
        kernel_size = get_eca_kernel_size(channels, gamma=gamma, b=b)
        padding = (kernel_size - 1) // 2

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class ECAWrapper(torch.nn.Module):
    """Wrapper used during training: base module followed by real ECA."""
    def __init__(self, base_module=None, gamma=2, b=1, *args, **kwargs):
        super().__init__()
        self.base = base_module
        self.gamma = gamma
        self.b = b
        self.eca = None
        self._initialized = False

        for attr in ("i", "f", "type", "np"):
            if self.base is not None and hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))

    def forward(self, x):
        if self.base is None:
            raise RuntimeError("ECAWrapper membutuhkan base module asli dari checkpoint")

        x = self.base(x)
        if not torch.is_tensor(x):
            return x

        eca = self._modules.get("eca") or self._modules.get("eca_layer")
        if eca is not None:
            self.eca = eca
            self._initialized = True
            return eca(x)

        channels = x.shape[1]
        self.eca = ECA(channels, gamma=self.gamma, b=self.b).to(x.device)
        self.add_module("eca_layer", self.eca)
        self._initialized = True
        return self.eca(x)


class AAttn(torch.nn.Module):
    """Compatibility shell for older Area Attention checkpoints."""
    def __init__(self, *args, **kwargs):
        super().__init__()

class MultiHeadAttention(torch.nn.Module):
    """Multi-Head Attention module"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        raise RuntimeError("MultiHeadAttention asli tidak tersedia di checkpoint ini")

class ScaledDotProductAttention(torch.nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        raise RuntimeError("ScaledDotProductAttention asli tidak tersedia di checkpoint ini")

class AdaptiveAttention(torch.nn.Module):
    """Adaptive Attention module"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        raise RuntimeError("AdaptiveAttention asli tidak tersedia di checkpoint ini")

# Register globally in builtins
import builtins
for cls in [CBAMWrapper, CBAM, ChannelAttention, SpatialAttention, ECA, ECAWrapper, AAttn,
            MultiHeadAttention, ScaledDotProductAttention, AdaptiveAttention]:
    builtins.__dict__[cls.__name__] = cls

sys.modules['__main__'].__dict__['CBAMWrapper'] = CBAMWrapper
sys.modules['__main__'].__dict__['CBAM'] = CBAM
sys.modules['__main__'].__dict__['ChannelAttention'] = ChannelAttention
sys.modules['__main__'].__dict__['SpatialAttention'] = SpatialAttention
sys.modules['__main__'].__dict__['ECA'] = ECA
sys.modules['__main__'].__dict__['ECAWrapper'] = ECAWrapper
sys.modules['__main__'].__dict__['AAttn'] = AAttn
sys.modules['__main__'].__dict__['MultiHeadAttention'] = MultiHeadAttention
sys.modules['__main__'].__dict__['ScaledDotProductAttention'] = ScaledDotProductAttention
sys.modules['__main__'].__dict__['AdaptiveAttention'] = AdaptiveAttention

# Custom Unpickler for handling CBAM modules
class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles CBAM and persistent hooks"""
    
    def find_class(self, module, name):
        """Override find_class to handle custom modules"""
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # Map notebook custom class names to their real API implementations.
            custom_classes = {
                'CBAMWrapper': CBAMWrapper,
                'CBAM': CBAM,
                'ChannelAttention': ChannelAttention,
                'SpatialAttention': SpatialAttention,
                'ECA': ECA,
                'ECAWrapper': ECAWrapper,
                'AAttn': AAttn,
                'MultiHeadAttention': MultiHeadAttention,
                'ScaledDotProductAttention': ScaledDotProductAttention,
                'AdaptiveAttention': AdaptiveAttention,
            }
            
            if name in custom_classes:
                return custom_classes[name]

            if "ECA" in name:
                return ECAWrapper if "Wrapper" in name else ECA
            
            # Try checking if it contains attention-related keywords
            if any(keyword in name for keyword in ['Attention', 'Attn', 'CBAM']):
                return AAttn if 'AAttn' in name else CBAMWrapper
            
            raise
    
    def persistent_load(self, pid):
        """Handle persistent_load for custom hooks"""
        return None

# Patch torch.load with persistent_load support
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Patched torch.load with persistent_load support"""
    try:
        return _original_torch_load(f, *args, **kwargs)
    except Exception as e:
        error_text = str(e).lower()
        if 'persistent' in error_text or 'cbam' in error_text or 'eca' in error_text or "can't get attribute" in error_text:
            if isinstance(f, (str, os.PathLike)):
                with open(f, 'rb') as file:
                    unpickler = SafeUnpickler(file)
                    return unpickler.load()
            else:
                return SafeUnpickler(f).load()
        raise

torch.load = patched_torch_load


def patch_ultralytics_attention_forward() -> None:
    """Patch newer Ultralytics AAttn forward to support older qk/v checkpoints."""
    try:
        from ultralytics.nn.modules import block as yolo_block
    except Exception as e:
        logging.getLogger(__name__).warning(f"Gagal patch Ultralytics AAttn: {e}")
        return

    if getattr(yolo_block.AAttn, "_legacy_qk_compat", False):
        return

    original_forward = yolo_block.AAttn.forward

    def legacy_compatible_forward(self, x):
        if "qkv" in self._modules:
            return original_forward(self, x)

        required = {"qk", "v", "pe", "proj"}
        if not required.issubset(self._modules):
            return original_forward(self, x)

        b, c, h, w = x.shape
        n = h * w
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(b * self.area, n // self.area, c * 2)
            v = v.reshape(b * self.area, n // self.area, c)
            b, n, _ = qk.shape

        q, k = qk.split([c, c], dim=2)
        q = q.transpose(1, 2).view(b, self.num_heads, self.head_dim, n)
        k = k.transpose(1, 2).view(b, self.num_heads, self.head_dim, n)
        v = v.transpose(1, 2).view(b, self.num_heads, self.head_dim, n)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(b // self.area, n * self.area, c)
            b, n, _ = x.shape

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return self.proj(x + pp)

    yolo_block.AAttn.forward = legacy_compatible_forward
    yolo_block.AAttn._legacy_qk_compat = True


patch_ultralytics_attention_forward()
import torch
import builtins

# # ==================================================
# # Dummy CBAM Classes
# # ==================================================

# class ChannelAttention(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(self, x):
#         return x


# class SpatialAttention(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(self, x):
#         return x


# class CBAM(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(self, x):
#         return x


# class CBAMWrapper(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(self, x):
#         return x


# Register to builtins
builtins.CBAM = CBAM
builtins.CBAMWrapper = CBAMWrapper
builtins.ChannelAttention = ChannelAttention
builtins.SpatialAttention = SpatialAttention
from ultralytics import YOLO

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
CBAM_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(BASE_DIR / "runs" / "detect" / "train4" / "weights" / "best.pt"),
)
ECA_MODEL_PATH = (
    os.getenv("YOLO_ECA_MODEL_PATH")
    or os.getenv("YOLO_ACA_MODEL_PATH")
    or os.getenv("YOLO_CBAM_MODEL_PATH")
    or r"D:\UMNFIX\yolov12_aca\runs\detect\train4\weights\best.pt"
)
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

ENABLE_EYE_VALIDATION = os.getenv("ENABLE_EYE_VALIDATION", "true").lower() != "false"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class Detection(BaseModel):
    """Model untuk hasil deteksi individual"""
    class_id: int = Field(..., description="ID kelas penyakit")
    class_name: str = Field(..., description="Nama penyakit mata")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: Dict[str, float] = Field(..., description="Bounding box (x1, y1, x2, y2)")
    bbox_normalized: Dict[str, float] = Field(..., description="Normalized bbox")
    area_pixels: int = Field(..., description="Luas deteksi dalam pixel")

class InferenceResponse(BaseModel):
    """Model response untuk hasil inference"""
    success: bool
    message: str
    timestamp: str
    image_metadata: Dict[str, Any]
    detections: List[Detection]
    statistics: Dict[str, Any]
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Optional evaluation metrics when ground truth is provided")
    image_base64: Optional[str] = Field(None, description="Annotated image in base64")

class BatchInferenceResponse(BaseModel):
    """Model response untuk batch inference"""
    success: bool
    message: str
    timestamp: str
    total_images: int
    processed_images: int
    results: List[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Optional aggregated evaluation metrics when ground truths are provided")

class HealthResponse(BaseModel):
    """Model untuk health check"""
    status: str
    model_loaded: bool
    device: str
    model_info: Dict[str, Any]
    timestamp: str

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================
app = FastAPI(
    title="YOLO v12 Eye Disease Detection API",
    description="""
    API untuk Deteksi Otomatis Penyakit Mata Anterior pada Citra Non-Fundus
    menggunakan YOLOv12 berbasis Attention dan Explainable AI
    
    **Fitur:**
    - Deteksi penyakit mata anterior dari citra non-fundus
    - Multiple image format support (JPEG, PNG, BMP, WebP)
    - Batch processing untuk multiple images
    - Explainable AI dengan confidence scores dan heatmaps
    - Comprehensive logging dan error handling
    
    **Penyakit yang dapat dideteksi:**
    Lihat endpoint `/v1/model/info` untuk daftar lengkap kelas penyakit
    """,
    version="2.0.0",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL MODEL INSTANCE
# ============================================================================
class ModelManager:
    """Manager untuk lifecycle model YOLO v12"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"  # Default to CPU
        self.is_loaded = False
        self.class_names = {}
        self.load_error = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load model YOLO v12"""
        try:
            logger.info(f"Loading YOLO v12 model dari: {self.model_path}")
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file tidak ditemukan: {self.model_path}")
            
            # Load model dengan device auto-detection
            try:
                # Try normal load
                self.model = YOLO(self.model_path)
            except Exception as e:
                error_msg = str(e)
                if "CBAMWrapper" in error_msg or "can't get attribute" in error_msg.lower() or "persistent" in error_msg.lower():
                    logger.warning(f"⚠ Custom module pickle error, trying alternative load...")
                    # Retry with patched torch.load already active
                    try:
                        self.model = YOLO(self.model_path)
                    except:
                        # Last attempt - force CPU mode
                        os.environ['YOLO_DEVICE'] = 'cpu'
                        self.model = YOLO(self.model_path)
                else:
                    raise
            
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                self.model.to(self.device)
            except:
                self.device = "cpu"
                logger.warning("⚠ Gagal move model ke GPU, menggunakan CPU")
            
            # Extract class names dari model
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            self.is_loaded = True
            logger.info(f"✓ Model berhasil dimuat pada device: {self.device}")
            logger.info(f"✓ Jumlah kelas: {len(self.class_names)}")
            logger.info(f"✓ Kelas-kelas: {self.class_names}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Error loading model: {str(e)}")
            logger.error(f"⚠ Model akan tetap tidak tersedia untuk inference")
            self.is_loaded = False
            self.load_error = str(e)
            self.device = "cpu"  # Always set device even on error
            # Set default empty classes
            self.class_names = {0: "Unknown"}
            return False
    
    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """Run inference pada image"""
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat")
        
        try:
            results = self.model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.filename:
        return False
    
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    
    return True

def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load image dari bytes"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

def _load_cascade(filename: str) -> Optional["cv2.CascadeClassifier"]:
    try:
        path = os.path.join(cv2.data.haarcascades, filename)
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            logger.warning(f"Cascade kosong / gagal dimuat: {filename}")
            return None
        return cascade
    except Exception as e:
        logger.warning(f"Gagal memuat cascade {filename}: {e}")
        return None


_EYE_CASCADE = _load_cascade("haarcascade_eye.xml")
_EYE_GLASSES_CASCADE = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
_FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")


def is_eye_image(image: np.ndarray) -> bool:
    cascades = [c for c in (_EYE_CASCADE, _EYE_GLASSES_CASCADE, _FACE_CASCADE) if c is not None]
    if not cascades:
        logger.warning("Tidak ada Haar cascade tersedia, melewati validasi mata")
        return True

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    except Exception as e:
        logger.warning(f"Gagal preprocessing untuk validasi mata: {e}")
        return True

    for cascade in cascades:
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(detections) > 0:
            return True

    return False


def convert_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array image to base64 string"""
    try:
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return ""
# ============================================================================
# GRAD-CAM FUNCTIONS
# ============================================================================


def generate_gradcam(model_manager, image: np.ndarray):
    """
    Generate Grad-CAM heatmap secara aman untuk YOLOv12 Custom Attention (CBAM/ECA)
    """
    try:
        # BGR -> RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_img_float = np.float32(rgb_img) / 255.0

        # TO TENSOR
        input_tensor = torch.from_numpy(
            rgb_img_float.transpose(2, 0, 1)
        ).unsqueeze(0).float()

        if model_manager.device == "cuda":
            input_tensor = input_tensor.cuda()

        # ====================================================================
        # DYNAMIC TARGET LAYER SELECTION (AMERIKA / ANTI-CRASH)
        # ====================================================================
        try:
            # Ambil model inti dari wrapper Ultralytics
            yolo_internal_model = model_manager.model.model
            
            # Cari layer konvolusi terakhir atau layer deteksi yang valid secara dinamis
            # Jika indexing [-6] bermasalah, kita pakai fallback aman ke layer konvolusi representatif
            target_layers = [yolo_internal_model.model[-4]] # Menggeser target layer ke area backbone/neck aman
        except Exception as layer_err:
            logger.warning(f"Gagal resolve dynamic target layer, menggunakan standard fallback: {layer_err}")
            # Jika tetap gagal menstrukturkan layer, bypass Grad-CAM agar inferensi utama box tetap jalan
            return image.copy()
        
        # ====================================================================
        # INIT CAM WITH EXCEPTION PROTECTION
        # ====================================================================
        try:
            cam = GradCAM(
                model=yolo_internal_model,
                target_layers=target_layers
            )
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor)[0]
        except Exception as cam_exec_err:
            # Proteksi jika kalkulasi forward/backward pass Grad-CAM crash karena channel mismatch di layer tersebut
            logger.error(f"GradCAM Execution Failed (Channel mismatch bypass active): {cam_exec_err}")
            return image.copy()

        # CREATE HEATMAP
        heatmap = show_cam_on_image(
            rgb_img_float,
            grayscale_cam,
            use_rgb=True
        )

        # RGB -> BGR
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        # BLEND
        visualization = cv2.addWeighted(
            image,
            0.5,
            heatmap_bgr,
            0.5,
            0
        )

        return visualization

    except Exception as e:
        logger.error(f"GradCAM Global Error: {str(e)}")
        return image.copy()


def draw_detections(
    image: np.ndarray,
    detections,
    class_names: Dict,
    model_manager
) -> np.ndarray:

    """
    Draw:
    - GradCAM heatmap
    - Bounding boxes
    - Labels
    """

    # =========================
    # GENERATE HEATMAP
    # =========================

    annotated = generate_gradcam(
        model_manager,
        image
    )

    if detections is None or not hasattr(detections, 'boxes'):
        return annotated

    boxes = detections.boxes

    if len(boxes) == 0:
        return annotated

    # =========================
    # DRAW BBOX
    # =========================

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        cls_name = class_names.get(
            cls_id,
            f"Class {cls_id}"
        )

        # Label
        label = f"{cls_name}: {conf:.3f}"

        # =========================
        # BOX COLOR
        # =========================

        color = (0, 255, 0)

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # =========================
        # TEXT BACKGROUND
        # =========================

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )

        # =========================
        # TEXT
        # =========================

        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return annotated

def calculate_detection_statistics(detections, image_shape) -> Dict[str, Any]:
    """Calculate statistics dari hasil deteksi"""
    stats = {
        "total_detections": 0,
        "average_confidence": 0.0,
        "min_confidence": 1.0,
        "max_confidence": 0.0,
        "image_area_pixels": image_shape[0] * image_shape[1],
        "detection_density": 0.0,
    }
    
    if detections is None or not hasattr(detections, 'boxes'):
        return stats
    
    boxes = detections.boxes
    if len(boxes) == 0:
        return stats
    
    confidences = [float(box.conf[0]) for box in boxes]
    stats["total_detections"] = len(boxes)
    stats["average_confidence"] = float(np.mean(confidences))
    stats["min_confidence"] = float(np.min(confidences))
    stats["max_confidence"] = float(np.max(confidences))
    stats["detection_density"] = stats["total_detections"] / (stats["image_area_pixels"] / 10000)
    
    return stats


def extract_detection_records(detections, class_names: Dict, image_shape) -> List[Dict[str, Any]]:
    """Convert model boxes into normalized detection records."""
    records = []

    if detections is None or not hasattr(detections, 'boxes'):
        return records

    h, w = image_shape[:2]

    for box in detections.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf_val = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, f"Unknown")
        area = int((x2 - x1) * (y2 - y1))

        records.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf_val,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "bbox_normalized": {
                "x1": x1 / w,
                "y1": y1 / h,
                "x2": x2 / w,
                "y2": y2 / h,
            },
            "area_pixels": area,
        })

    return records


def build_detection_models(records: List[Dict[str, Any]]) -> List[Detection]:
    """Convert detection record dictionaries into response models."""
    return [Detection(**record) for record in records]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _normalize_bbox(raw_bbox: Dict[str, Any]) -> Dict[str, float]:
    """Validate and normalize bbox payload."""
    if not isinstance(raw_bbox, dict):
        raise ValueError("bbox harus berupa object dengan x1, y1, x2, y2")

    required_keys = {"x1", "y1", "x2", "y2"}
    if not required_keys.issubset(raw_bbox):
        raise ValueError("bbox harus berisi x1, y1, x2, y2")

    bbox = {key: float(raw_bbox[key]) for key in required_keys}
    if bbox["x2"] <= bbox["x1"] or bbox["y2"] <= bbox["y1"]:
        raise ValueError("bbox ground truth tidak valid")

    return bbox


def _normalize_ground_truth_payload(payload: Any) -> Dict[str, Any]:
    """Normalize a ground-truth payload into a consistent structure."""
    if payload is None:
        return {"objects": [], "iou_threshold": 0.5}

    if isinstance(payload, list):
        raw_objects = payload
        iou_threshold = 0.5
    elif isinstance(payload, dict):
        raw_objects = payload.get("objects", payload.get("detections"))
        if raw_objects is None:
            raise ValueError("ground truth harus punya field 'objects' atau 'detections'")
        iou_threshold = float(payload.get("iou_threshold", 0.5))
    else:
        raise ValueError("ground truth harus berupa JSON object atau array")

    if not isinstance(raw_objects, list):
        raise ValueError("field 'objects' harus berupa array")

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold ground truth harus di antara 0 dan 1")

    objects = []
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, dict):
            raise ValueError(f"ground truth object ke-{index} harus berupa object")
        if "class_id" not in raw_object:
            raise ValueError(f"ground truth object ke-{index} wajib punya class_id")
        if "bbox" not in raw_object:
            raise ValueError(f"ground truth object ke-{index} wajib punya bbox")

        objects.append({
            "class_id": int(raw_object["class_id"]),
            "class_name": raw_object.get("class_name"),
            "bbox": _normalize_bbox(raw_object["bbox"]),
        })

    return {"objects": objects, "iou_threshold": iou_threshold}


def parse_ground_truth_json(raw_ground_truth: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse optional ground-truth JSON from multipart form data."""
    if raw_ground_truth is None or not raw_ground_truth.strip():
        return None

    try:
        payload = json.loads(raw_ground_truth)
        return _normalize_ground_truth_payload(payload)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"ground_truth bukan JSON valid: {e.msg}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"ground_truth tidak valid: {str(e)}") from e


def parse_batch_ground_truths(raw_ground_truths: Optional[str], files: List[UploadFile]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Parse batch ground truths from multipart form data.

    Supported formats:
    - JSON list aligned with urutan `files`
    - JSON object keyed by filename
    """
    if raw_ground_truths is None or not raw_ground_truths.strip():
        return {}

    try:
        payload = json.loads(raw_ground_truths)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"ground_truths bukan JSON valid: {e.msg}") from e

    try:
        if isinstance(payload, list):
            if len(payload) != len(files):
                raise ValueError("ground_truths array harus sama panjang dengan jumlah files")
            return {
                file.filename or f"file_{idx}": _normalize_ground_truth_payload(item)
                for idx, (file, item) in enumerate(zip(files, payload))
            }

        if isinstance(payload, dict):
            normalized = {}
            for idx, file in enumerate(files):
                key = file.filename or f"file_{idx}"
                item = payload.get(key)
                normalized[key] = _normalize_ground_truth_payload(item) if item is not None else None
            return normalized

        raise ValueError("ground_truths harus berupa JSON object atau array")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"ground_truths tidak valid: {str(e)}") from e


def calculate_iou(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
    """Calculate IoU between two bounding boxes."""
    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])
    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    inter_width = max(0.0, x_right - x_left)
    inter_height = max(0.0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area_a = max(0.0, (box_a["x2"] - box_a["x1"])) * max(0.0, (box_a["y2"] - box_a["y1"]))
    area_b = max(0.0, (box_b["x2"] - box_b["x1"])) * max(0.0, (box_b["y2"] - box_b["y1"]))
    union_area = area_a + area_b - inter_area

    return _safe_ratio(inter_area, union_area)


def evaluate_detection_records(
    predictions: List[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Any]],
    class_names: Dict[int, str],
) -> Optional[Dict[str, Any]]:
    """Compute precision/recall/F1 from predictions and optional ground truth."""
    if ground_truth is None:
        return None

    iou_threshold = float(ground_truth["iou_threshold"])
    gt_objects = ground_truth["objects"]
    matched_gt = set()
    sorted_predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)

    tp = 0
    fp = 0
    per_class_counts: Dict[int, Dict[str, int]] = {}

    def ensure_class_bucket(class_id: int) -> Dict[str, int]:
        if class_id not in per_class_counts:
            per_class_counts[class_id] = {
                "ground_truth_count": 0,
                "prediction_count": 0,
                "true_positives": 0,
                "false_positives": 0,
            }
        return per_class_counts[class_id]

    for gt in gt_objects:
        ensure_class_bucket(int(gt["class_id"]))["ground_truth_count"] += 1

    for pred in sorted_predictions:
        pred_class_id = int(pred["class_id"])
        ensure_class_bucket(pred_class_id)["prediction_count"] += 1

        best_match_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_objects):
            if gt_idx in matched_gt or int(gt["class_id"]) != pred_class_id:
                continue
            current_iou = calculate_iou(pred["bbox"], gt["bbox"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_match_idx = gt_idx

        if best_match_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_match_idx)
            tp += 1
            ensure_class_bucket(pred_class_id)["true_positives"] += 1
        else:
            fp += 1
            ensure_class_bucket(pred_class_id)["false_positives"] += 1

    fn = len(gt_objects) - len(matched_gt)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_score = _safe_ratio(2 * precision * recall, precision + recall)
    mean_confidence = float(np.mean([item["confidence"] for item in predictions])) if predictions else 0.0

    per_class_metrics = {}
    for class_id, counts in sorted(per_class_counts.items()):
        class_tp = counts["true_positives"]
        class_fp = counts["false_positives"]
        class_fn = counts["ground_truth_count"] - class_tp
        class_precision = _safe_ratio(class_tp, class_tp + class_fp)
        class_recall = _safe_ratio(class_tp, class_tp + class_fn)
        class_f1 = _safe_ratio(2 * class_precision * class_recall, class_precision + class_recall)
        class_name = class_names.get(class_id, f"Class {class_id}")

        per_class_metrics[class_name] = {
            "class_id": class_id,
            "class_name": class_name,
            "ground_truth_count": counts["ground_truth_count"],
            "prediction_count": counts["prediction_count"],
            "true_positives": class_tp,
            "false_positives": class_fp,
            "false_negatives": class_fn,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
        }

    return {
        "iou_threshold": iou_threshold,
        "ground_truth_count": len(gt_objects),
        "prediction_count": len(predictions),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_confidence": mean_confidence,
        "per_class": per_class_metrics,
    }


def aggregate_evaluations(evaluations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate per-image evaluations into one batch summary."""
    valid_evaluations = [item for item in evaluations if item]
    if not valid_evaluations:
        return None

    tp = sum(item["true_positives"] for item in valid_evaluations)
    fp = sum(item["false_positives"] for item in valid_evaluations)
    fn = sum(item["false_negatives"] for item in valid_evaluations)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_score = _safe_ratio(2 * precision * recall, precision + recall)

    thresholds = {item["iou_threshold"] for item in valid_evaluations}
    per_class_totals: Dict[str, Dict[str, Any]] = {}

    for item in valid_evaluations:
        for class_name, metrics in item.get("per_class", {}).items():
            bucket = per_class_totals.setdefault(class_name, {
                "class_id": metrics["class_id"],
                "class_name": metrics["class_name"],
                "ground_truth_count": 0,
                "prediction_count": 0,
                "true_positives": 0,
                "false_positives": 0,
            })
            bucket["ground_truth_count"] += metrics["ground_truth_count"]
            bucket["prediction_count"] += metrics["prediction_count"]
            bucket["true_positives"] += metrics["true_positives"]
            bucket["false_positives"] += metrics["false_positives"]

    for class_name, metrics in per_class_totals.items():
        class_fn = metrics["ground_truth_count"] - metrics["true_positives"]
        class_precision = _safe_ratio(metrics["true_positives"], metrics["true_positives"] + metrics["false_positives"])
        class_recall = _safe_ratio(metrics["true_positives"], metrics["true_positives"] + class_fn)
        class_f1 = _safe_ratio(2 * class_precision * class_recall, class_precision + class_recall)
        metrics["false_negatives"] = class_fn
        metrics["precision"] = class_precision
        metrics["recall"] = class_recall
        metrics["f1_score"] = class_f1

    return {
        "images_evaluated": len(valid_evaluations),
        "iou_threshold": next(iter(thresholds)) if len(thresholds) == 1 else None,
        "ground_truth_count": sum(item["ground_truth_count"] for item in valid_evaluations),
        "prediction_count": sum(item["prediction_count"] for item in valid_evaluations),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_confidence": _safe_ratio(
            sum(item["mean_confidence"] * item["prediction_count"] for item in valid_evaluations),
            sum(item["prediction_count"] for item in valid_evaluations),
        ),
        "per_class": per_class_totals,
    }

# ============================================================================
# INITIALIZE MODELS
# ============================================================================
try:
    cbam_model_manager = ModelManager(CBAM_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize CBAM model: {str(e)}")
    cbam_model_manager = None

try:
    eca_model_manager = ModelManager(ECA_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize ECA model: {str(e)}")
    eca_model_manager = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/v1/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    """Health check endpoint untuk kedua model (CBAM dan ECA)"""
    models_status = {}
    
    if cbam_model_manager:
        models_status["cbam"] = {
            "loaded": cbam_model_manager.is_loaded,
            "device": cbam_model_manager.device,
            "num_classes": len(cbam_model_manager.class_names),
        }
    
    if eca_model_manager:
        models_status["eca"] = {
            "loaded": eca_model_manager.is_loaded,
            "device": eca_model_manager.device,
            "num_classes": len(eca_model_manager.class_names),
        }
    
    all_loaded = (cbam_model_manager and cbam_model_manager.is_loaded) or (eca_model_manager and eca_model_manager.is_loaded)
    
    return HealthResponse(
        status="healthy" if all_loaded else "unhealthy",
        model_loaded=all_loaded,
        device=cbam_model_manager.device if cbam_model_manager else (eca_model_manager.device if eca_model_manager else "unknown"),
        model_info=models_status,
        timestamp=datetime.now().isoformat()
    )

@app.get("/v1/model/info", tags=["Model Information"])
async def model_info():
    """Get detailed information tentang semua model yang tersedia"""
    models_info = {}
    
    # CBAM Model Info
    if cbam_model_manager and cbam_model_manager.is_loaded:
        models_info["cbam"] = {
            "name": "YOLOv12 with CBAM",
            "task": "Object Detection",
            "attention": "CBAM (Convolutional Block Attention Module)",
            "path": CBAM_MODEL_PATH,
            "device": cbam_model_manager.device,
            "num_classes": len(cbam_model_manager.class_names),
            "classes": cbam_model_manager.class_names,
        }
    
    # ECA Model Info
    if eca_model_manager and eca_model_manager.is_loaded:
        models_info["eca"] = {
            "name": "YOLOv12 ACA with ECA",
            "task": "Object Detection",
            "attention": "ECA (Efficient Channel Attention)",
            "path": ECA_MODEL_PATH,
            "device": eca_model_manager.device,
            "num_classes": len(eca_model_manager.class_names),
            "classes": eca_model_manager.class_names,
        }
    
    if not models_info:
        raise HTTPException(status_code=503, detail="Tidak ada model yang tersedia")
    
    return {
        "success": True,
        "models": models_info,
        "research": {
            "title": "DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12 BERBASIS ATTENTION DAN EXPLAINABLE AI",
            "description": "Automatic detection of anterior eye diseases from non-fundus images using YOLOv12 with attention mechanisms",
            "framework": "YOLOv12",
            "enhancements": [
                "CBAM (Convolutional Block Attention Module)",
                "ECA (Efficient Channel Attention)",
                "Legacy Attention (AAttn)",
            ],
            "interpretability": "Grad-CAM for explainable AI",
        },
        "endpoints": {
            "cbam": {
                "detect": "/v1/detect",
                "batch": "/v1/detect/batch",
            },
            "eca": {
                "detect": "/v1/detect/eca",
                "batch": "/v1/detect/eca/batch",
            }
        },
        "supported_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
        "api_version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/detect", response_model=InferenceResponse, tags=["CBAM Inference"])
async def detect_disease(
    file: UploadFile = File(..., description="Image file untuk dianalisis dengan model CBAM"),
    conf: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.1, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=1.0, description="IOU threshold"),
    return_image: bool = Query(False, description="Return annotated image in base64"),
    ground_truth: Optional[str] = Form(
        None,
        description='Optional JSON evaluation payload, example: {"objects":[{"class_id":0,"bbox":{"x1":10,"y1":20,"x2":100,"y2":120}}],"iou_threshold":0.5}'
    )
):
    """
    Deteksi penyakit mata menggunakan model YOLOv12 dengan CBAM
    
    **Parameters:**
    - `file`: Image file (JPEG, PNG, BMP, WebP)
    - `conf`: Confidence threshold (default: 0.25)
    - `iou`: IOU threshold untuk NMS (default: 0.45)
    - `return_image`: Include annotated image dalam response (base64)

    **Model:** YOLOv12 CBAM dari D:\\UMNFIX\\yolov12\\runs\\detect\\train4\\weights\\best.pt
    """
    
    if not cbam_model_manager or not cbam_model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="CBAM Model tidak tersedia")

    parsed_ground_truth = parse_ground_truth_json(ground_truth)
    
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="File format tidak didukung")
    
    try:
        # Read file
        contents = await file.read()
        
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="File terlalu besar (max 50MB)")
        
        # Load image
        image = load_image_from_bytes(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Gagal membaca image")

        if ENABLE_EYE_VALIDATION and not is_eye_image(image):
            raise HTTPException(
                status_code=422,
                detail="Gambar tidak terdeteksi sebagai citra mata. Unggah foto mata yang valid."
            )

        # Save temporarily - use proper temp directory for cross-platform compatibility
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name

        cv2.imwrite(temp_path, image)

        # Run inference
        results = cbam_model_manager.predict(temp_path, conf=conf, iou=iou)
        
        detection_records = extract_detection_records(results, cbam_model_manager.class_names, image.shape)
        detections_list = build_detection_models(detection_records)
        evaluation = evaluate_detection_records(
            detection_records,
            parsed_ground_truth,
            cbam_model_manager.class_names,
        )
        
        # Get statistics
        stats = calculate_detection_statistics(results, image.shape)
        
        # Draw annotations
        annotated_image = draw_detections(
    image=image,
    detections=results,
    class_names=cbam_model_manager.class_names,
    model_manager=cbam_model_manager
)
        # Encode image if requested
        image_base64 = None
        if return_image:
            image_base64 = convert_image_to_base64(annotated_image)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return InferenceResponse(
            success=True,
            message=f"CBAM Deteksi selesai: {len(detections_list)} penyakit ditemukan",
            timestamp=datetime.now().isoformat(),
            image_metadata={
                "filename": file.filename,
                "size": len(contents),
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "model": "YOLOv12 CBAM",
            },
            detections=detections_list,
            statistics=stats,
            evaluation=evaluation,
            image_base64=image_base64
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/v1/detect/batch", response_model=BatchInferenceResponse, tags=["CBAM Batch Processing"])
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    conf: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.1, le=1.0),
    iou: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=1.0),
    ground_truths: Optional[str] = Form(
        None,
        description='Optional JSON evaluation payloads. Use a filename-keyed object or an array aligned with files.'
    ),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Batch inference dengan model CBAM untuk multiple images
    
    **Parameters:**
    - `files`: List of image files
    - `conf`: Confidence threshold
    - `iou`: IOU threshold
    """
    
    if not cbam_model_manager or not cbam_model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="CBAM Model tidak tersedia")
    
    if len(files) > 50:
        raise HTTPException(status_code=413, detail="Maksimal 50 images per batch")

    parsed_ground_truths = parse_batch_ground_truths(ground_truths, files)
    
    results = []
    processed_count = 0
    collected_evaluations = []
    
    for idx, file in enumerate(files):
        try:
            file_key = file.filename or f"file_{idx}"
            if not validate_image_file(file):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Format tidak didukung"
                })
                continue
            
            contents = await file.read()
            image = load_image_from_bytes(contents)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Gagal membaca image"
                })
                continue

            if ENABLE_EYE_VALIDATION and not is_eye_image(image):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Gambar tidak terdeteksi sebagai citra mata"
                })
                continue

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix or ".png", delete=False) as tmp_file:
                temp_path = tmp_file.name
            cv2.imwrite(temp_path, image)

            result = cbam_model_manager.predict(temp_path, conf=conf, iou=iou)
            detection_records = extract_detection_records(result, cbam_model_manager.class_names, image.shape)
            evaluation = evaluate_detection_records(
                detection_records,
                parsed_ground_truths.get(file_key),
                cbam_model_manager.class_names,
            )
            
            # Count detections
            detection_count = len(detection_records)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detection_count,
                "image_shape": {"width": image.shape[1], "height": image.shape[0]},
                "model": "YOLOv12 CBAM",
                "evaluation": evaluation,
            })
            
            processed_count += 1
            if evaluation:
                collected_evaluations.append(evaluation)
            background_tasks.add_task(os.remove, temp_path)
        
        except Exception as e:
            logger.error(f"Batch processing error for {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return BatchInferenceResponse(
        success=True,
        message=f"CBAM Batch processing selesai: {processed_count}/{len(files)} berhasil",
        timestamp=datetime.now().isoformat(),
        total_images=len(files),
        processed_images=processed_count,
        results=results,
        evaluation=aggregate_evaluations(collected_evaluations)
    )

@app.post("/v1/detect/eca", response_model=InferenceResponse, tags=["ECA Inference"])
async def detect_disease_eca(
    file: UploadFile = File(..., description="Image file untuk dianalisis dengan ECA model"),
    conf: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.1, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=1.0, description="IOU threshold"),
    return_image: bool = Query(False, description="Return annotated image in base64"),
    ground_truth: Optional[str] = Form(
        None,
        description='Optional JSON evaluation payload, example: {"objects":[{"class_id":0,"bbox":{"x1":10,"y1":20,"x2":100,"y2":120}}],"iou_threshold":0.5}'
    )
):
    """
    Deteksi penyakit mata menggunakan YOLOv12 ACA dengan ECA (Efficient Channel Attention)
    
    **Parameters:**
    - `file`: Image file (JPEG, PNG, BMP, WebP)
    - `conf`: Confidence threshold (default: 0.25)
    - `iou`: IOU threshold untuk NMS (default: 0.45)
    - `return_image`: Include annotated image dalam response (base64)
    
    **Model:** YOLOv12 ACA dengan ECA dari D:\\UMNFIX\\yolov12_aca\\runs\\detect\\train4\\weights\\best.pt
    """
    
    if not eca_model_manager or not eca_model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="ECA Model tidak tersedia")

    parsed_ground_truth = parse_ground_truth_json(ground_truth)
    
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="File format tidak didukung")
    
    try:
        # Read file
        contents = await file.read()
        
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="File terlalu besar (max 50MB)")
        
        # Load image
        image = load_image_from_bytes(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Gagal membaca image")

        if ENABLE_EYE_VALIDATION and not is_eye_image(image):
            raise HTTPException(
                status_code=422,
                detail="Gambar tidak terdeteksi sebagai citra mata. Unggah foto mata yang valid."
            )

        # Save temporarily - use proper temp directory for cross-platform compatibility
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name

        cv2.imwrite(temp_path, image)

        # Run inference dengan ECA model
        results = eca_model_manager.predict(temp_path, conf=conf, iou=iou)
        
        detection_records = extract_detection_records(results, eca_model_manager.class_names, image.shape)
        detections_list = build_detection_models(detection_records)
        evaluation = evaluate_detection_records(
            detection_records,
            parsed_ground_truth,
            eca_model_manager.class_names,
        )
        
        # Get statistics
        stats = calculate_detection_statistics(results, image.shape)
        
        # Draw annotations
        annotated_image = draw_detections(image, results, eca_model_manager.class_names, eca_model_manager)
        
        # Encode image if requested
        image_base64 = None
        if return_image:
            image_base64 = convert_image_to_base64(annotated_image)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return InferenceResponse(
            success=True,
            message=f"ECA Deteksi selesai: {len(detections_list)} penyakit ditemukan",
            timestamp=datetime.now().isoformat(),
            image_metadata={
                "filename": file.filename,
                "size": len(contents),
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "model": "YOLOv12 ECA",
            },
            detections=detections_list,
            statistics=stats,
            evaluation=evaluation,
            image_base64=image_base64
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ECA Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ECA Inference error: {str(e)}")

@app.post("/v1/detect/eca/batch", response_model=BatchInferenceResponse, tags=["ECA Batch Processing"])
async def detect_batch_eca(
    files: List[UploadFile] = File(..., description="Multiple image files untuk ECA model"),
    conf: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.1, le=1.0),
    iou: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=1.0),
    ground_truths: Optional[str] = Form(
        None,
        description='Optional JSON evaluation payloads. Use a filename-keyed object or an array aligned with files.'
    ),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Batch inference dengan ECA model untuk multiple images
    
    **Parameters:**
    - `files`: List of image files
    - `conf`: Confidence threshold
    - `iou`: IOU threshold
    
    **Model:** YOLOv12 ACA dengan ECA
    """
    
    if not eca_model_manager or not eca_model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="ECA Model tidak tersedia")
    
    if len(files) > 50:
        raise HTTPException(status_code=413, detail="Maksimal 50 images per batch")

    parsed_ground_truths = parse_batch_ground_truths(ground_truths, files)
    
    results = []
    processed_count = 0
    collected_evaluations = []
    
    for idx, file in enumerate(files):
        try:
            file_key = file.filename or f"file_{idx}"
            if not validate_image_file(file):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Format tidak didukung"
                })
                continue
            
            contents = await file.read()
            image = load_image_from_bytes(contents)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Gagal membaca image"
                })
                continue

            if ENABLE_EYE_VALIDATION and not is_eye_image(image):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Gambar tidak terdeteksi sebagai citra mata"
                })
                continue

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix or ".png", delete=False) as tmp_file:
                temp_path = tmp_file.name
            cv2.imwrite(temp_path, image)

            result = eca_model_manager.predict(temp_path, conf=conf, iou=iou)
            detection_records = extract_detection_records(result, eca_model_manager.class_names, image.shape)
            evaluation = evaluate_detection_records(
                detection_records,
                parsed_ground_truths.get(file_key),
                eca_model_manager.class_names,
            )
            
            # Count detections
            detection_count = len(detection_records)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detection_count,
                "image_shape": {"width": image.shape[1], "height": image.shape[0]},
                "model": "YOLOv12 ECA",
                "evaluation": evaluation,
            })
            
            processed_count += 1
            if evaluation:
                collected_evaluations.append(evaluation)
            background_tasks.add_task(os.remove, temp_path)
        
        except Exception as e:
            logger.error(f"ECA Batch processing error for {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return BatchInferenceResponse(
        success=True,
        message=f"ECA Batch processing selesai: {processed_count}/{len(files)} berhasil",
        timestamp=datetime.now().isoformat(),
        total_images=len(files),
        processed_images=processed_count,
        results=results,
        evaluation=aggregate_evaluations(collected_evaluations)
    )

@app.get("/v1/supported-formats", tags=["Information"])
async def supported_formats():
    """Get list of supported image formats"""
    return {
        "supported_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
        "max_file_size_mb": MAX_IMAGE_SIZE // (1024 * 1024),
        "max_batch_size": 50
    }

# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint dengan informasi API"""
    return {
        "name": "YOLO v12 Eye Disease Detection API",
        "version": "2.0.0",
        "research": "Deteksi Otomatis Penyakit Mata Anterior pada Citra Non-Fundus menggunakan YOLOv12",
        "documentation": "/v1/docs",
        "health": "/v1/health",
        "endpoints": {
            "health_check": "/v1/health",
            "model_info": "/v1/model/info",
            "cbam_model": {
                "detect": "/v1/detect",
                "batch_detect": "/v1/detect/batch",
            },
            "eca_model": {
                "detect": "/v1/detect/eca",
                "batch_detect": "/v1/detect/eca/batch",
            },
            "supported_formats": "/v1/supported-formats",
        },
        "available_models": ["cbam (YOLOv12)", "eca (YOLOv12 ACA)"],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("=" * 80)
    logger.info("YOLO v12 Eye Disease Detection API - Starting")
    logger.info("=" * 80)
    
    if cbam_model_manager and cbam_model_manager.is_loaded:
        logger.info("✓ CBAM Model loaded successfully")
    else:
        logger.warning("✗ CBAM Model failed to load")
    
    if eca_model_manager and eca_model_manager.is_loaded:
        logger.info("✓ ECA Model loaded successfully")
    else:
        logger.warning("✗ ECA Model failed to load")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("YOLO v12 Eye Disease Detection API - Shutting down")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
