import os
import sys
import torch
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir / "ultralytics_backup"))
sys.path.insert(0, str(current_dir))

# Memasukkan direktori aktif ke sys.path agar modul lokal terbaca
sys.path.append(os.getcwd())

# Ubah nama file di bawah ini sesuai yang ingin kamu tes
MODEL_NAME = "best_clean.pt" 
MODEL_PATH = Path("runs/detect/train4/weights") / MODEL_NAME

print("=" * 60)
print(f"MEMULAI TEST LOAD UNTUK: {MODEL_PATH}")
print("=" * 60)

# 1. Cek Fisik File
if not MODEL_PATH.exists():
    print(f"❌ ERROR: File {MODEL_PATH} TIDAK DITEMUKAN!")
    sys.exit(1)

file_size = MODEL_PATH.stat().st_size / (1024 * 1024)
print(f"✓ File ditemukan. Ukuran fisik: {file_size:.2f} MB")

# 2. Tes Load Menggunakan PyTorch Native (Mendeteksi Isu Pickle)
print("\n[Mulai] Tes 1: Loading menggunakan torch.load()...")
try:
    # Menggunakan map_location='cpu' karena kita running di MacBook (ARM)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    print("✅ SUKSES: PyTorch native berhasil membaca struktur file!")
    
    # Intip isi keys di dalam checkpoint .pt
    if isinstance(checkpoint, dict):
        print(f"   Keys yang tersedia di dalam file: {list(checkpoint.keys())}")
    else:
        print(f"   Tipe data model ter-load: {type(checkpoint)}")
except Exception as e:
    print(f"❌ GAGAL di Tes 1 (PyTorch): {e}")

# 3. Tes Load Menggunakan High-Level Framework YOLO Ultralytics
print("\n[Mulai] Tes 2: Loading menggunakan Ultralytics YOLO framework...")
try:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print("✅ SUKSES: Framework Ultralytics YOLO berhasil menginisialisasi model!")
    print(f"   Informasi Model: {model.info()}")
except Exception as e:
    print(f"❌ GAGAL di Tes 2 (Ultralytics): {e}")

print("\n" + "=" * 60)
print("TEST SELESAI")
print("=" * 60)