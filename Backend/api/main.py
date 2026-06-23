"""
Backend/api/main.py — FastAPI inference server for ECG Alarm Fatigue Reduction
v2.0 — XGBoost (AUC 0.93) as primary + CNN (AUC 0.756) as secondary (ensemble)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import os, pickle, tempfile
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'src', 'models')

# ── CNN architecture (unchanged) ───────────────────────────────────────────
class AlarmCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout), nn.MaxPool1d(4))
        self.block2 = nn.Sequential(nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout), nn.MaxPool1d(4))
        self.block3 = nn.Sequential(nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout), nn.MaxPool1d(4))
        self.block4 = nn.Sequential(nn.Conv1d(128,256,3,padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout), nn.AdaptiveAvgPool1d(8))
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(256*8, 128)
        self.bn_fc   = nn.BatchNorm1d(128)
        self.relu_fc = nn.ReLU()
        self.drop_fc = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, 2)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.flatten(x)
        return self.fc2(self.drop_fc(self.relu_fc(self.bn_fc(self.fc1(x)))))

# ── Load CNN ───────────────────────────────────────────────────────────────
device    = torch.device('cpu')
cnn_model = AlarmCNN().to(device)
cnn_model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'alarm_cnn_final.pt'), map_location=device))
cnn_model.eval()

# ── Load XGBoost (primary model) ───────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'xgboost_model.pkl'), 'rb') as f:
    xgb_obj   = pickle.load(f)
xgb_model = xgb_obj['model']
xgb_feats = xgb_obj['features']   # exact feature list model was trained on

# ── Load Random Forest (optional secondary tabular model) ──────────────────
rf_model  = None
rf_feats  = None
rf_path   = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
if os.path.exists(rf_path):
    with open(rf_path, 'rb') as f:
        rf_obj = pickle.load(f)
    rf_model = rf_obj['model']
    rf_feats = rf_obj.get('features', xgb_feats)

# ── Load metadata ──────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'model_metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

TARGET_LEN = metadata.get('target_len', 9375)
ML_AUC     = metadata.get('ml_auc', {})
CNN_AUC    = metadata.get('cnn_auc', 0.756)

print(f"✅ CNN + XGBoost{' + RandomForest' if rf_model else ''} loaded | target_len={TARGET_LEN}")

# ── FastAPI setup ──────────────────────────────────────────────────────────
app = FastAPI(title="ECG Alarm Fatigue API", version="2.0.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)

# ── Signal helpers ─────────────────────────────────────────────────────────
def bandpass(sig, fs=250):
    nyq  = 0.5 * fs
    b, a = butter(3, [0.5/nyq, min(40/nyq, 0.99)], btype='band')
    return filtfilt(b, a, sig)

def detect_rpeaks(ecg, fs=250):
    """Simple R-peak detector — matches training notebook exactly."""
    sq     = ecg ** 2
    win    = int(0.15 * fs)
    smooth = np.convolve(sq, np.ones(win)/win, mode='same')
    thresh = 0.6 * np.percentile(smooth, 98)
    above  = smooth > thresh
    cross  = np.where(np.diff(above.astype(int)) == 1)[0]
    ref    = int(0.4 * fs)
    peaks, last = [], -ref
    for c in cross:
        if c - last >= ref:
            s = max(0, c - win//2)
            e = min(len(ecg), c + win//2)
            p = s + np.argmax(sq[s:e])
            peaks.append(p)
            last = p
    return np.array(peaks)

def preprocess_cnn(sig, fs=250):
    """Preprocess raw signal for CNN input — same as original main.py."""
    s   = pd.Series(sig.astype(float))
    sig = s.interpolate(method='linear', limit_direction='both').values
    sig = bandpass(sig, fs)
    if len(sig) >= TARGET_LEN:
        sig = sig[-TARGET_LEN:]
    else:
        sig = np.pad(sig, (TARGET_LEN - len(sig), 0), mode='edge')
    m, sd = sig.mean(), sig.std()
    if sd > 1e-8:
        sig = (sig - m) / sd
    return sig.astype(np.float32)

def extract_xgb_features(signals, sig_names, fs=250):
    """
    Extract the 29 tabular features XGBoost was trained on.
    Must match 03_feature_extraction.ipynb exactly.
    """
    # Find channels by name
    def get_ch(keywords):
        for kw in keywords:
            for i, n in enumerate(sig_names):
                if kw.lower() in str(n).lower():
                    return signals[:, i]
        return None

    ecg   = get_ch(['II', 'ECG', 'I']) or signals[:, 0]
    pleth = get_ch(['PLETH', 'SPO2', 'pleth'])
    abp   = get_ch(['ABP', 'BP', 'abp'])

    # Clean + filter ECG
    ecg_s = pd.Series(ecg.astype(float)).interpolate(
        method='linear', limit_direction='both').values
    ecg_f = bandpass(ecg_s, fs)

    # Whole-signal features
    feats = {
        'has_pleth'   : int(pleth is not None),
        'has_abp'     : int(abp   is not None),
        'n_channels'  : signals.shape[1],
        'ecg_mean'    : float(np.nanmean(ecg_f)),
        'ecg_std'     : float(np.nanstd(ecg_f)),
        'ecg_kurtosis': float(kurtosis(ecg_f)),
        'ecg_skew'    : float(skew(ecg_f)),
        'ecg_energy'  : float(np.nanmean(ecg_f ** 2)),
    }

    # R-peak / HR / RR features
    peaks = detect_rpeaks(ecg_f, fs)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs * 1000   # ms
        hr = 60000 / rr
        feats.update({
            'hr_mean'        : float(np.mean(hr)),
            'hr_std'         : float(np.std(hr)),
            'hr_min'         : float(np.min(hr)),
            'hr_max'         : float(np.max(hr)),
            'rr_mean'        : float(np.mean(rr)),
            'rr_std'         : float(np.std(rr)),
            'rmssd'          : float(np.sqrt(np.mean(np.diff(rr) ** 2))),
            'n_peaks'        : float(len(peaks)),
            'short_rr_ratio' : float(np.mean(rr < 300)),
            'long_rr_ratio'  : float(np.mean(rr > 1500)),
        })
    else:
        for k in ['hr_mean','hr_std','hr_min','hr_max','rr_mean',
                  'rr_std','rmssd','n_peaks','short_rr_ratio','long_rr_ratio']:
            feats[k] = 0.0

    # Last-30s window features
    win       = ecg_f[-int(30 * fs):]
    win_peaks = detect_rpeaks(win, fs)
    feats['win_ecg_std']    = float(np.std(win))
    feats['win_ecg_energy'] = float(np.mean(win ** 2))
    if len(win_peaks) >= 2:
        win_rr = np.diff(win_peaks) / fs * 1000
        feats['win_hr_mean']        = float(np.mean(60000 / win_rr))
        feats['win_rr_std']         = float(np.std(win_rr))
        feats['win_short_rr_ratio'] = float(np.mean(win_rr < 300))
        feats['win_long_rr_ratio']  = float(np.mean(win_rr > 1500))
    else:
        feats['win_hr_mean']        = feats['hr_mean']
        feats['win_rr_std']         = feats['rr_std']
        feats['win_short_rr_ratio'] = feats['short_rr_ratio']
        feats['win_long_rr_ratio']  = feats['long_rr_ratio']

    # PLETH features
    if pleth is not None:
        pl = pd.Series(pleth.astype(float)).interpolate(
            limit_direction='both').values
        feats['pleth_mean'] = float(np.nanmean(pl))
        feats['pleth_std']  = float(np.nanstd(pl))
    else:
        feats['pleth_mean'] = 0.0
        feats['pleth_std']  = 0.0

    # ABP features
    if abp is not None:
        ab = pd.Series(abp.astype(float)).interpolate(
            limit_direction='both').values
        feats['abp_mean'] = float(np.nanmean(ab))
        feats['abp_std']  = float(np.nanstd(ab))
        feats['abp_pp']   = float(np.nanmax(ab) - np.nanmin(ab))
    else:
        feats['abp_mean'] = 0.0
        feats['abp_std']  = 0.0
        feats['abp_pp']   = 0.0

    return feats

def load_mat(data: bytes):
    """Load a .mat file and return (signals_2d, sig_names, fs)."""
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        mat = loadmat(path)
        # PhysioNet 2015 format stores signals in 'val'
        if 'val' in mat:
            sig = mat['val']
            # val shape is (n_channels, n_samples) — transpose to (n_samples, n_channels)
            if sig.ndim == 1:
                sig = sig.reshape(-1, 1)
            elif sig.shape[0] < sig.shape[1]:
                sig = sig.T
            return sig.astype(float), [f'ch{i}' for i in range(sig.shape[1])], 250

        # Fallback: find first array
        for k, v in mat.items():
            if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim >= 1:
                sig = v if v.ndim == 2 else v.reshape(-1, 1)
                if sig.shape[0] < sig.shape[1]:
                    sig = sig.T
                return sig.astype(float), [f'ch{i}' for i in range(sig.shape[1])], 250

        raise ValueError("No signal array found in .mat file")
    finally:
        os.unlink(path)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "version": "2.0 — XGBoost + CNN ensemble"}

@app.get("/model-comparison")
def model_comparison():
    return {"models": [
        {"name": "Logistic Regression", "auc": round(ML_AUC.get("logistic_regression", 0.766), 3), "std": 0.059},
        {"name": "Random Forest",       "auc": round(ML_AUC.get("random_forest",       0.835), 3), "std": 0.072},
        {"name": "XGBoost",             "auc": round(ML_AUC.get("xgboost",             0.825), 3), "std": 0.075},
        {"name": "CNN (fine-tuned)",    "auc": round(CNN_AUC,                                  3), "std": 0.033},
    ]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.mat'):
        raise HTTPException(400, "Only .mat files are supported")

    # ── Read file ──────────────────────────────────────────────────────────
    try:
        data                    = await file.read()
        signals, sig_names, fs  = load_mat(data)
    except Exception as e:
        raise HTTPException(422, f"Cannot read .mat file: {e}")

    try:
        # ── Tabular feature extraction (shared by XGBoost / Random Forest) ─
        feats   = extract_xgb_features(signals, sig_names, fs)
        feat_df = pd.DataFrame([feats])

        # XGBoost (primary — AUC ~0.825)
        xgb_in     = feat_df[xgb_feats]
        p_true_xgb = float(xgb_model.predict_proba(xgb_in)[0][1])

        # Random Forest (secondary tabular model — AUC ~0.835), if available
        if rf_model is not None:
            rf_in     = feat_df[rf_feats]
            p_true_rf = float(rf_model.predict_proba(rf_in)[0][1])
        else:
            p_true_rf = p_true_xgb

        # ── CNN inference (AUC ~0.756) ──────────────────────────────────────
        ecg_raw = signals[:, 0]
        sig_cnn = preprocess_cnn(ecg_raw, fs)
        t       = torch.FloatTensor(sig_cnn).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_probs = torch.softmax(cnn_model(t), dim=1).squeeze().tolist()
        p_true_cnn = float(cnn_probs[1])

        # ── Weighted ensemble for final prediction ─────────────────────────
        # Weight by relative AUC quality (XGB and RF strong, CNN weaker)
        p_true  = 0.45 * p_true_xgb + 0.35 * p_true_rf + 0.20 * p_true_cnn
        p_false = 1.0 - p_true

        is_true    = p_true >= 0.5
        confidence = max(p_true, p_false)

        if confidence >= 0.80:
            confidence_label = "HIGH"
        elif confidence >= 0.60:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        # ── Alarm type heuristic from heart-rate features ──────────────────
        hr_mean = feats.get('hr_mean', 0.0)
        if hr_mean <= 0:
            alarm_type = "Unknown"
        elif hr_mean > 130:
            alarm_type = "Tachycardia"
        elif hr_mean < 50:
            alarm_type = "Bradycardia"
        elif feats.get('short_rr_ratio', 0) > 0.05 or feats.get('long_rr_ratio', 0) > 0.05:
            alarm_type = "Arrhythmia"
        else:
            alarm_type = "Other"

        waveform = sig_cnn[::16].tolist()   # downsample 9375→~586 for display

        return {
            "prediction"      : "TRUE ALARM" if is_true else "FALSE ALARM",
            "is_true_alarm"   : is_true,
            "true_prob"       : round(p_true, 4),
            "false_prob"      : round(p_false, 4),
            "confidence"      : round(confidence, 4),
            "confidence_label": confidence_label,
            "alarm_type"      : alarm_type,
            "model_scores": {
                "xgboost"      : round(p_true_xgb, 4),
                "random_forest": round(p_true_rf, 4),
                "cnn"          : round(p_true_cnn, 4),
            },
            "waveform": waveform,
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")