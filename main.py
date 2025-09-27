import io
import os
import tempfile
import time
import zipfile
from typing import List, Tuple
import base64

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
import imageio.v2 as iio
import pydicom
import pandas as pd
from torchvision import models, transforms
import SimpleITK as sitk
from app_db import SessionLocal, Prediction, init_db
from pydantic import BaseModel


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(APP_ROOT, "outputs")
# Prefer model checkpoint from project root to avoid mounting large outputs
DEFAULT_CKPT = os.path.join(APP_ROOT, "final_model_200ct.pt")
FALLBACK_CKPT = os.path.join(APP_ROOT, "resnet18_regularized.pt")

init_db()


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


 


def _uids_from_file(path: str) -> Tuple[str, str]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return str(getattr(ds, 'StudyInstanceUID', '')), str(getattr(ds, 'SeriesInstanceUID', ''))
    except Exception:
        return '', ''


def read_series_sorted(files: List[str]) -> List[str]:
    """Sort DICOM files by InstanceNumber or ImagePositionPatient (z)."""
    def key_fn(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if hasattr(ds, 'InstanceNumber'):
                return int(ds.InstanceNumber)
            if hasattr(ds, 'ImagePositionPatient'):
                ipp = list(ds.ImagePositionPatient)
                return float(ipp[2]) if len(ipp) > 2 else 0.0
        except Exception:
            return 0
        return 0
    return sorted(files, key=key_fn)


def predict_from_zip_bytes(zip_bytes: bytes, max_slices: int = 64) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmpdir)
        files = read_dicom_series_from_dir(tmpdir)
        if not files:
            raise ValueError("No DICOM series found in the uploaded ZIP")
        study_uid, series_uid = _uids_from_file(files[0])
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
            "study_uid": study_uid,
            "series_uid": series_uid,
            "time_of_processing": round(time.time() - t0, 3),
        }


 


def write_report(records: List[dict], out_path: str):
    df = pd.DataFrame(records)
    # Map label to 0/1
    df['pathology'] = (df['label'] == 'abnormal').astype(int)
    df.rename(columns={
        'probability': 'probability_of_pathology',
        'study_uid': 'study_uid',
        'series_uid': 'series_uid',
        'time_of_processing': 'time_of_processing',
    }, inplace=True)
    # Ensure required columns
    if 'path_to_study' not in df.columns:
        df['path_to_study'] = ''
    if 'processing_status' not in df.columns:
        df['processing_status'] = 'Success'
    cols = [
        'path_to_study', 'study_uid', 'series_uid',
        'probability_of_pathology', 'pathology', 'processing_status', 'time_of_processing'
    ]
    df = df.reindex(columns=cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)


def append_report(record: dict, out_path: str):
    """Append single record to report.xlsx, creating it if missing."""
    # normalize single record using write_report schema
    existing = None
    if os.path.isfile(out_path):
        try:
            existing = pd.read_excel(out_path)
        except Exception:
            existing = None
    write_report([record], out_path + ".tmp.xlsx")
    new_df = pd.read_excel(out_path + ".tmp.xlsx")
    try:
        os.remove(out_path + ".tmp.xlsx")
    except Exception:
        pass
    if existing is not None and not existing.empty:
        all_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        all_df = new_df
    all_df.to_excel(out_path, index=False)


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

# Serve generated reports from outputs/
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
# Expose outputs directory for downloading generated reports
os.makedirs(OUTPUTS_DIR, exist_ok=True)
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


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
        # persist/update report incrementally
        result['path_to_study'] = file.filename
        result['processing_status'] = 'Success'
        append_report(result, os.path.join(OUTPUTS_DIR, 'report.xlsx'))
        # save to DB
        try:
            if SessionLocal is not None:
                with SessionLocal() as db:
                    rec = Prediction(
                        path_to_study=result.get('path_to_study',''),
                        study_uid=result.get('study_uid',''),
                        series_uid=result.get('series_uid',''),
                        probability_of_pathology=float(result.get('probability',0.0)),
                        pathology=1 if result.get('label') == 'abnormal' else 0,
                        processing_status=result.get('processing_status','Success'),
                        time_of_processing=float(result.get('time_of_processing',0.0)),
                    )
                    db.add(rec)
                    db.commit()
        except Exception:
            pass
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/batch")
async def batch(files: List[UploadFile] = File(...)):
    records: List[dict] = []
    for f in files:
        # Initialize default failure record
        rec = {
            'path_to_study': f.filename,
            'processing_status': 'Success',
            'probability': 0.0,
            'label': 'normal',
            'time_of_processing': 0.0,
            'study_uid': '',
            'series_uid': ''
        }
        if not f.filename.lower().endswith('.zip'):
            rec['processing_status'] = 'Failure: not a ZIP'
            records.append(rec)
            continue
        content = await f.read()
        try:
            res = predict_from_zip_bytes(content)
            res['path_to_study'] = f.filename
            res['processing_status'] = 'Success'
            records.append(res)
        except Exception as e:
            rec['processing_status'] = f'Failure: {str(e)}'
            records.append(rec)
    # Write report
    report_path = os.path.join(OUTPUTS_DIR, 'report.xlsx')
    write_report(records, report_path)
    # bulk insert
    try:
        if SessionLocal is not None and records:
            with SessionLocal() as db:
                for r in records:
                    rec = Prediction(
                        path_to_study=r.get('path_to_study',''),
                        study_uid=r.get('study_uid',''),
                        series_uid=r.get('series_uid',''),
                        probability_of_pathology=float(r.get('probability',0.0)),
                        pathology=1 if r.get('label') == 'abnormal' else 0,
                        processing_status=r.get('processing_status','Success'),
                        time_of_processing=float(r.get('time_of_processing',0.0)),
                    )
                    db.add(rec)
                db.commit()
    except Exception:
        pass
    return {
        'count': len(records),
        'report_url': '/outputs/report.xlsx',
        'records': records,
    }


@app.post("/series_pngs")
async def series_pngs(file: UploadFile = File(...), max_slices: int = 256, wl: float = -600.0, ww: float = 1500.0):
    if not file.filename.lower().endswith('.zip'):
        return JSONResponse(status_code=400, content={"error": "Please upload a ZIP with a DICOM study"})
    content = await file.read()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(tmpdir)
            reader = sitk.ImageSeriesReader()
            files = read_dicom_series_from_dir(tmpdir)
            if not files:
                return JSONResponse(status_code=400, content={"error": "No series found"})
            files = read_series_sorted(files)
            reader.SetFileNames(files)
            img = reader.Execute()
            img = to_3d(img)
            if img.GetDimension() != 3:
                return JSONResponse(status_code=400, content={"error": "Not a 3D series"})
            # Apply requested WL/WW
            img = window_lung(img, center=float(wl), width=float(ww))
            vol = sitk.GetArrayFromImage(img)  # (Z,Y,X) uint8
            z, y, x = vol.shape
            # Downsample slices if too many
            idxs = list(range(z)) if z <= max_slices else np.linspace(0, z-1, max_slices, dtype=int).tolist()
            pngs = []
            for i in idxs:
                arr = vol[i].astype(np.uint8)
                with io.BytesIO() as buf:
                    iio.imwrite(buf, arr, format='png')
                    pngs.append(base64.b64encode(buf.getvalue()).decode('ascii'))
            return {"count": len(pngs), "width": int(x), "height": int(y), "slices": pngs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ExportRequest(BaseModel):
    ids: List[int]


@app.get("/history")
async def history():
    items = []
    try:
        if SessionLocal is not None:
            with SessionLocal() as db:
                rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(500).all()
                for r in rows:
                    items.append({
                        'id': r.id,
                        'path_to_study': r.path_to_study,
                        'study_uid': r.study_uid,
                        'series_uid': r.series_uid,
                        'probability_of_pathology': r.probability_of_pathology,
                        'pathology': r.pathology,
                        'processing_status': r.processing_status,
                        'time_of_processing': r.time_of_processing,
                        'created_at': r.created_at.isoformat() if r.created_at else None,
                    })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"items": items}


@app.get("/history_page", response_class=HTMLResponse)
async def history_page(request: Request):
    items = []
    try:
        if SessionLocal is not None:
            with SessionLocal() as db:
                rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(500).all()
                for r in rows:
                    items.append({
                        'id': r.id,
                        'path_to_study': r.path_to_study,
                        'study_uid': r.study_uid,
                        'series_uid': r.series_uid,
                        'probability_of_pathology': r.probability_of_pathology,
                        'pathology': r.pathology,
                        'processing_status': r.processing_status,
                        'time_of_processing': r.time_of_processing,
                        'created_at': r.created_at.isoformat() if r.created_at else None,
                    })
    except Exception:
        items = []
    return templates.TemplateResponse("history.html", {"request": request, "items": items})


@app.get("/viewer", response_class=HTMLResponse)
async def viewer(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})


 


@app.post("/export_selected")
async def export_selected(req: ExportRequest):
    if not req.ids:
        return JSONResponse(status_code=400, content={"error": "No ids provided"})
    records = []
    try:
        if SessionLocal is not None:
            with SessionLocal() as db:
                rows = db.query(Prediction).filter(Prediction.id.in_(req.ids)).all()
                for r in rows:
                    records.append({
                        'path_to_study': r.path_to_study,
                        'study_uid': r.study_uid,
                        'series_uid': r.series_uid,
                        'probability': r.probability_of_pathology,
                        'label': 'abnormal' if r.pathology == 1 else 'normal',
                        'processing_status': r.processing_status,
                        'time_of_processing': r.time_of_processing,
                    })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    out_path = os.path.join(OUTPUTS_DIR, 'selected_report.xlsx')
    write_report(records, out_path)
    return {"count": len(records), "report_url": "/outputs/selected_report.xlsx"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


