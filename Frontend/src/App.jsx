import { useState, useRef, useCallback } from "react";
import "./App.css";

/* ─── tiny helpers ─────────────────────────────────────────────────────────── */
const pct = (n) => `${Math.round((n ?? 0) * 100)}%`;

/* ─── fixed training AUC values ───────────────────────────────────────────── */
const TRAINING_AUC = {
  xgboost: 0.825,
  random_forest: 0.835,
  cnn: 0.756,
};

/* ─── Donut ────────────────────────────────────────────────────────────────── */
function Donut({ value = 0, color, size = 140, centerLabel, centerSub }) {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const safe = Math.max(0, Math.min(1, value));
  const offset = circ * (1 - safe);
  return (
    <svg width={size} height={size} viewBox="0 0 100 100">
      <circle cx="50" cy="50" r={r} fill="none" stroke="#1b3a52" strokeWidth="11" />
      <circle
        cx="50" cy="50" r={r} fill="none"
        stroke={color} strokeWidth="11"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(-90 50 50)"
        style={{ transition: "stroke-dashoffset 0.9s ease" }}
      />
      {centerLabel && (
        <>
          <text x="50" y="46" textAnchor="middle" fontSize="11" fontWeight="700" fill={color} fontFamily="Arial">{centerLabel}</text>
          {centerSub && <text x="50" y="60" textAnchor="middle" fontSize="7" fill="#7ea8c0" fontFamily="Arial">{centerSub}</text>}
        </>
      )}
    </svg>
  );
}

/* ─── Gauge ────────────────────────────────────────────────────────────────── */
function Gauge({ value = 0, label = "LOW" }) {
  const safe   = Math.max(0, Math.min(1, value));
  const angle  = -135 + safe * 270;
  const color  = label === "HIGH" ? "#2dd4bf" : label === "MEDIUM" ? "#f59e0b" : "#ef4444";
  const arcLen = safe * 196;
  return (
    <div className="gauge-box">
      <svg width="160" height="100" viewBox="0 0 160 100">
        <path d="M18 90 A62 62 0 0 1 142 90" fill="none" stroke="#1b3a52" strokeWidth="12" strokeLinecap="round"/>
        <path d="M18 90 A62 62 0 0 1 142 90" fill="none" stroke={color} strokeWidth="12"
          strokeLinecap="round" strokeDasharray={`${arcLen} 196`}/>
        <g transform={`rotate(${angle} 80 90)`}>
          <line x1="80" y1="90" x2="80" y2="36" stroke="#cfe8f3" strokeWidth="3" strokeLinecap="round"/>
          <circle cx="80" cy="90" r="5" fill={color}/>
        </g>
      </svg>
      <p className="gauge-val" style={{ color }}>{pct(safe)} ({label})</p>
      <p className="gauge-lbl">MODEL CONFIDENCE</p>
    </div>
  );
}

/* ─── Waveform ─────────────────────────────────────────────────────────────── */
function Waveform({ data = [], isTrue }) {
  if (!Array.isArray(data) || data.length < 4) return (
    <div className="waveform-empty">No waveform data</div>
  );
  const W = 480, H = 130;
  const step    = Math.max(1, Math.floor(data.length / 600));
  const sampled = data.filter((_, i) => i % step === 0).map(Number);
  const min = Math.min(...sampled), max = Math.max(...sampled);
  const rng = max - min || 1;
  const norm = (v) => H - 8 - ((v - min) / rng) * (H - 16);
  const pts  = sampled.map((v, i) => `${((i / (sampled.length - 1)) * W).toFixed(1)},${norm(v).toFixed(1)}`).join(" ");
  const ax   = W * 0.78;
  const color = isTrue ? "#ef4444" : "#ef4444"; /* red highlight matches reference */

  return (
    <div className="waveform">
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {/* grid */}
        {[0.25, 0.5, 0.75].map(f => (
          <line key={f} x1="0" y1={H * f} x2={W} y2={H * f} stroke="#1b3a52" strokeWidth="0.6" strokeDasharray="5 5"/>
        ))}
        {[0.2, 0.4, 0.6, 0.8].map(f => (
          <line key={f} x1={W * f} y1="0" x2={W * f} y2={H} stroke="#1b3a52" strokeWidth="0.6" strokeDasharray="5 5"/>
        ))}
        {/* alarm zone highlight — orange-red tint like reference */}
        <rect x={ax} y="0" width={W - ax} height={H} fill="rgba(239,100,68,0.13)"/>
        {/* signal */}
        <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8"/>
        {/* alarm line */}
        <line x1={ax} y1="0" x2={ax} y2={H} stroke={color} strokeWidth="1" strokeDasharray="4 3"/>
        <text x={ax + 4} y="13" fill={color} fontSize="7" fontFamily="Arial">ALARM</text>
      </svg>
    </div>
  );
}

/* ─── Model Score Row ──────────────────────────────────────────────────────── */
function ScoreRow({ name, score = 0, color }) {
  const safe = Math.max(0, Math.min(1, score));
  return (
    <div className="score-row">
      <span className="score-name">{name}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${safe * 100}%`, background: color }}/>
      </div>
      <span className="score-val" style={{ color }}>{pct(safe)}</span>
    </div>
  );
}

/* ─── Main App ─────────────────────────────────────────────────────────────── */
export default function App() {
  const [file,    setFile]    = useState(null);
  const [drag,    setDrag]    = useState(false);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);
  const inputRef = useRef(null);

  /* active model tab state */
  const [activeTab, setActiveTab] = useState("xgboost");

  const pickFile = (f) => {
    if (!f?.name?.endsWith(".mat")) { setError("Please upload a .mat file."); return; }
    setFile(f); setError(null); setResult(null);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, []);

  const detect = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("http://localhost:8000/predict", { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `Error ${res.status}`); }
      const raw = await res.json();
      setResult({
        prediction:       raw.prediction      ?? "FALSE ALARM",
        true_prob:        Number(raw.true_prob  ?? raw.true_probability  ?? 0),
        false_prob:       Number(raw.false_prob ?? raw.false_probability ?? 0),
        confidence:       Number(raw.confidence ?? 0),
        confidence_label: raw.confidence_label ?? "LOW",
        alarm_type:       raw.alarm_type       ?? "Unknown",
        scores: {
          xgboost:       Number(raw.model_scores?.xgboost       ?? 0),
          random_forest: Number(raw.model_scores?.random_forest ?? 0),
          cnn:           Number(raw.model_scores?.cnn           ?? 0),
        },
        waveform: Array.isArray(raw.waveform) ? raw.waveform.map(Number) : [],
      });
    } catch (e) {
      setError(e.message || "Connection failed — is the backend running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const isTrue = result?.prediction === "TRUE ALARM";

  /* tab labels */
  const TABS = [
    { key: "xgboost",       label: "XGBOOST"       },
    { key: "random_forest", label: "RANDOM FOREST"  },
    { key: "cnn",           label: "CNN"            },
  ];

  return (
    <div className="app">

      {/* ── TOP HEADER ─────────────────────────────────────────────────────── */}
      <header className="top-bar">
        <div className="top-bar__left">
          <div className="logo-icon"><HeartIcon/></div>
          <span className="top-bar__title">𝗖𝗮𝗿𝗱𝗶𝗼𝗚𝗮𝘂𝗿𝗱-𝗔𝗜</span>️
        </div>
        {/* model tabs moved here just as decorative — actual panel is on right */}
        <div className="top-bar__tabs">
          <span className="tabs-label">MODEL PERFORMANCE</span>
          <div className="tab-strip">
            {TABS.map(t => (
              <button key={t.key}
                className={`tab-btn ${activeTab === t.key ? "tab-btn--on" : ""}`}
                onClick={() => setActiveTab(t.key)}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── SUB HEADER ─────────────────────────────────────────────────────── */}
      <div className="sub-bar">
        <div className="sub-bar__brand">
          <div className="sub-bar__icon"><HeartIcon/></div>
          <div>
            <p className="sub-bar__name">CARDIAC ALARM VERIFIER</p>
            <p className="sub-bar__desc">Medical AI application for your medical dashboard</p>
          </div>
        </div>
      </div>

      {/* ── MAIN GRID ──────────────────────────────────────────────────────── */}
      <main className="grid">

        {/* LEFT — upload */}
        <section className="card card--upload">
          <h2 className="card__title">
            <UploadIcon/> Upload .MAT File
          </h2>

          <div
            className={`drop ${drag ? "drop--over" : ""} ${file ? "drop--filled" : ""}`}
            onDragOver={(e)=>{ e.preventDefault(); setDrag(true); }}
            onDragLeave={()=>setDrag(false)}
            onDrop={onDrop}
            onClick={()=>inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept=".mat" hidden
              onChange={e=>e.target.files?.[0] && pickFile(e.target.files[0])}/>
            <UploadBigIcon/>
            <p className="drop__text">Drag and drop file</p>
            <p className="drop__or">or</p>
            <button
              type="button"
              className="btn-browse"
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
            >
              BROWSE FILE
            </button>
          </div>

          {file && (
            <div className="file-pill">
              <span>{file.name}</span>
              <button onClick={()=>{ setFile(null); setResult(null); }}>×</button>
            </div>
          )}

          {error && <p className="err">{error}</p>}

          <button
            className={`btn-detect ${(!file || loading) ? "btn-detect--off" : ""}`}
            disabled={!file || loading}
            onClick={detect}>
            {loading ? <span className="spin"/> : <><SearchIcon/> DETECT ALARM</>}
          </button>
        </section>

        {/* CENTRE — results */}
        <section className="card card--results">
          <h2 className="card__title"><SearchIcon/> DETECTION RESULTS</h2>

          {!result ? (
            <div className="results-empty">
              <FlatHeartIcon/>
              <p>Upload a .mat file and click Detect Alarm to see results</p>
            </div>
          ) : (
            <div className="results-body">
              {/* waveform */}
              <Waveform data={result.waveform} isTrue={isTrue}/>

              {/* metrics row */}
              <div className="metrics-row">
                {/* donut */}
                <div className="donut-col">
                  <Donut
                    value={result.true_prob}
                    color="#f59e0b"
                    size={148}
                    centerLabel="TRUE ALARM"
                    centerSub="PROBABILITY"
                  />
                </div>

                {/* text probs */}
                <div className="prob-col">
                  <p className="prob-lbl">TRUE ALARM PROBABILITY:</p>
                  <p className="prob-val" style={{color:"#f59e0b"}}>{pct(result.true_prob)}</p>
                  <p className="prob-lbl" style={{marginTop:14}}>FALSE ALARM PROBABILITY:</p>
                  <p className="prob-val" style={{color:"#2dd4bf"}}>{pct(result.false_prob)}</p>
                </div>

                {/* prediction badge */}
                <div className="badge-col">
                  <div className={`pred-badge ${isTrue ? "pred-badge--true" : "pred-badge--false"}`}>
                    <span className="pred-badge__top">PREDICTION:</span>
                    <span className="pred-badge__val">{result.prediction}</span>
                  </div>
                </div>
              </div>

              {/* gauge */}
              <div className="gauge-row">
                <Gauge value={result.confidence} label={result.confidence_label}/>
              </div>

              {/* alarm type */}
              <div className="alarm-tag">
                <span>ALARM TYPE</span>
                <span className="alarm-tag__val">{result.alarm_type.replace(/_/g," ")}</span>
              </div>
            </div>
          )}
        </section>

        {/* RIGHT — model performance panel (replaces session history) */}
        <section className="card card--models">
          <h2 className="card__title"><ChartIcon/> MODEL PERFORMANCE</h2>

          {/* tab switcher inside panel */}
          <div className="panel-tabs">
            {TABS.map(t => (
              <button key={t.key}
                className={`ptab ${activeTab === t.key ? "ptab--on" : ""}`}
                onClick={()=>setActiveTab(t.key)}>
                {t.label}
              </button>
            ))}
          </div>

          {/* active model detail card */}
          <div className="model-detail">
            {activeTab === "xgboost" && <ModelCard
              name="XGBoost"
              full="eXtreme Gradient Boosting"
              color="#10b981"
              auc={TRAINING_AUC.xgboost}
              desc="Builds trees sequentially, each correcting prior errors. Uses scale_pos_weight to handle class imbalance. Best for tabular feature data."
              features={["Gradient boosting trees","Early stopping","Feature importance via gain"]}
            />}
            {activeTab === "random_forest" && <ModelCard
              name="Random Forest"
              full="Ensemble of Decision Trees"
              color="#2dd4bf"
              auc={TRAINING_AUC.random_forest}
              desc="300 trees each trained on random subsets. Majority vote reduces variance. Best ML performer on this dataset."
              features={["300 estimators","Balanced class weights","Out-of-bag evaluation"]}
            />}
            {activeTab === "cnn" && <ModelCard
              name="CNN"
              full="1D Convolutional Neural Network"
              color="#6366f1"
              auc={TRAINING_AUC.cnn}
              desc="Pre-trained on 87k MIT-BIH heartbeat segments, fine-tuned on 748 alarm records. Learns directly from raw waveform."
              features={["4 conv blocks","Transfer learning","5-fold cross-validation"]}
            />}
          </div>

          {/* score comparison bars — always visible */}
          <div className="scores-section">
            <p className="scores-section__title">LIVE SCORES {result ? "(current file)" : "(training AUC)"}</p>
            <ScoreRow name="XGBoost"       score={result ? result.scores.xgboost       : TRAINING_AUC.xgboost}       color="#10b981"/>
            <ScoreRow name="Random Forest" score={result ? result.scores.random_forest : TRAINING_AUC.random_forest} color="#2dd4bf"/>
            <ScoreRow name="CNN"           score={result ? result.scores.cnn           : TRAINING_AUC.cnn}           color="#6366f1"/>
          </div>

          {/* training AUC reference */}
          <div className="auc-ref">
            <p className="auc-ref__title">5-FOLD CV TRAINING AUC</p>
            <div className="auc-chips">
              <div className="auc-chip"><span style={{color:"#94a3b8"}}>LR</span> <strong style={{color:"#e2e8f0"}}>0.766</strong></div>
              <div className="auc-chip"><span style={{color:"#10b981"}}>XGB</span><strong style={{color:"#e2e8f0"}}>0.825</strong></div>
              <div className="auc-chip"><span style={{color:"#2dd4bf"}}>RF</span> <strong style={{color:"#e2e8f0"}}>0.835</strong></div>
            </div>
          </div>
        </section>
      </main>

      {/* ── FOOTER ─────────────────────────────────────────────────────────── */}
      <footer className="foot">
        <nav className="foot__nav">
          <a href="#">About</a>
          <a href="#">Documentation</a>
          <a href="#">Support</a>
        </nav>
        <p className="foot__copy">
          Disclaimer: © 2024 CARDIAC ALARM dashboard — Research prototype. Not for clinical use.
        </p>
      </footer>
    </div>
  );
}

/* ─── Model Card sub-component ─────────────────────────────────────────────── */
function ModelCard({ name, full, color, auc, desc, features }) {
  return (
    <div className="mcard">
      <div className="mcard__top">
        <div>
          <p className="mcard__name" style={{color}}>{name}</p>
          <p className="mcard__full">{full}</p>
        </div>
        <div className="mcard__auc">
          <span className="mcard__auc-val" style={{color}}>{auc > 0 ? auc.toFixed(3) : "—"}</span>
          <span className="mcard__auc-lbl">AUC</span>
        </div>
      </div>
      <p className="mcard__desc">{desc}</p>
      <ul className="mcard__feats">
        {features.map(f => <li key={f}><span style={{color}}>▸</span> {f}</li>)}
      </ul>
    </div>
  );
}

/* ─── SVG icons ────────────────────────────────────────────────────────────── */
function HeartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  );
}
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
    </svg>
  );
}
function UploadBigIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="#2dd4bf" strokeWidth="1.5" width="40" height="40">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
    </svg>
  );
}
function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="15" height="15">
      <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
    </svg>
  );
}
function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  );
}
function FlatHeartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="#253d52" strokeWidth="1.5" width="52" height="52">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  );
}