import io
import os
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
from torchvision import models, transforms
import SimpleITK as sitk


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(APP_ROOT, "outputs")
DEFAULT_CKPT = os.path.join(OUTPUTS_DIR, "final_model_200ct.pt")
FALLBACK_CKPT = os.path.join(OUTPUTS_DIR, "resnet18_regularized.pt")


def to_3d(image: sitk.Image, t_index: int = 0) -> sitk.Image:
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        index = [0, 0, 0, 0]
        index[3] = t_index
        size[3] = 0
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(image)
    return image


def window_lung(image: sitk.Image, center: float = -600.0, width: float = 1500.0) -> sitk.Image:
    image = sitk.Cast(image, sitk.sitkFloat32)
    min_v, max_v = center - width / 2.0, center + width / 2.0
    win = sitk.IntensityWindowing(image, float(min_v), float(max_v), 0.0, 255.0)
    return sitk.Cast(win, sitk.sitkUInt8)


def build_model(dropout: float = 0.5) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


def load_checkpoint() -> Tuple[nn.Module, torch.device, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt_path = DEFAULT_CKPT if os.path.isfile(DEFAULT_CKPT) else FALLBACK_CKPT
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {DEFAULT_CKPT} or {FALLBACK_CKPT}")
    # PyTorch >=2.6 defaults to weights_only=True which breaks legacy checkpoints
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = ckpt.get("params", {})
    dropout = float(params.get("dropout", 0.5))
    model = build_model(dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, device, {"checkpoint": os.path.basename(ckpt_path), "params": params}


MODEL, DEVICE, MODEL_META = load_checkpoint()


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_dicom_series_from_dir(root_dir: str) -> List[str]:
    """Recursively find a DICOM series and return the file list for the first detected series."""
    reader = sitk.ImageSeriesReader()
    # try root first
    try:
        sids = reader.GetGDCMSeriesIDs(root_dir) or []
    except Exception:
        sids = []
    if sids:
        return reader.GetGDCMSeriesFileNames(root_dir, sids[0])
    # recurse
    for dirpath, dirnames, _ in os.walk(root_dir):
        try:
            sids = reader.GetGDCMSeriesIDs(dirpath) or []
        except Exception:
            sids = []
        if sids:
            return reader.GetGDCMSeriesFileNames(dirpath, sids[0])
    return []


def predict_from_zip_bytes(zip_bytes: bytes, max_slices: int = 64) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmpdir)
        files = read_dicom_series_from_dir(tmpdir)
        if not files:
            raise ValueError("No DICOM series found in the uploaded ZIP")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        image = reader.Execute()
        image = to_3d(image)
        if image.GetDimension() != 3:
            raise ValueError("Could not obtain a 3D volume from the uploaded DICOMs")
        image8 = window_lung(image)
        vol = sitk.GetArrayFromImage(image8)  # (Z, Y, X)
        z = vol.shape[0]
        idxs = np.linspace(0, z - 1, min(max_slices, z), dtype=int)

        slice_probs: List[float] = []
        with torch.no_grad():
            for i in idxs:
                arr = vol[i].astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
                x = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
                logit = MODEL(x)
                prob = torch.sigmoid(logit).item()
                slice_probs.append(float(prob))

        study_prob = float(np.mean(slice_probs)) if slice_probs else 0.0
        label = "abnormal" if study_prob > 0.5 else "normal"
        return {
            "probability": study_prob,
            "label": label,
            "num_slices": int(len(slice_probs)),
            "checkpoint": MODEL_META["checkpoint"],
            "params": MODEL_META.get("params", {}),
        }


app = FastAPI(title="CT Normal/Abnormal Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = os.path.join(APP_ROOT, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "checkpoint": MODEL_META["checkpoint"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "Please upload a ZIP file with a DICOM study"})
    content = await file.read()
    try:
        result = predict_from_zip_bytes(content)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


