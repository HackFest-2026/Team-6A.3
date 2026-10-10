import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  FileImage,
  HeartPulse,
  LoaderCircle,
  ScanLine,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { api, type HistoryItem, type Prediction } from './api'

const maxBytes = 5 * 1024 * 1024
const classes = ['Healthy', 'Not Healthy', 'Pneumonia']

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [apiReady, setApiReady] = useState(false)

  useEffect(() => {
    void api.get('/health').then(() => setApiReady(true)).catch(() => setApiReady(false))
    void api.get<HistoryItem[]>('/history').then(({ data }) => setHistory(data)).catch(() => setHistory([]))
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview('')
      return
    }
    const objectUrl = URL.createObjectURL(file)
    setPreview(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  function acceptFile(candidate?: File) {
    if (!candidate) return
    setError('')
    setPrediction(null)
    const isDicom = /\.(dcm|dicom)$/i.test(candidate.name)
    if (!isDicom && !['image/jpeg', 'image/png'].includes(candidate.type)) {
      setError('Choose a JPG, JPEG, PNG, or DICOM image.')
      return
    }
    if (candidate.size > maxBytes) {
      setError('This image is over the 5 MB limit.')
      return
    }
    setFile(candidate)
  }

  async function runPrediction() {
    if (!file) return
    setBusy(true)
    setError('')
    const body = new FormData()
    body.append('file', file)
    try {
      const { data } = await api.post<Prediction>('/predict', body)
      setPrediction(data)
      const historyItem: HistoryItem = {
        id: data.id,
        filename: data.filename,
        predicted_class: data.predicted_class,
        confidence: data.probabilities[data.predicted_class],
        created_at: data.created_at,
        inference_ms: data.inference_ms,
      }
      setHistory((items) => [historyItem, ...items.filter((item) => item.id !== data.id)].slice(0, 20))
    } catch (caught) {
      const message = axios.isAxiosError(caught)
        ? caught.response?.data?.detail ?? 'Unable to reach the analysis service. Check that the API and model checkpoints are available.'
        : 'The image could not be analyzed. Try again.'
      setError(typeof message === 'string' ? message : 'The image could not be analyzed.')
    } finally {
      setBusy(false)
    }
  }

  async function clearHistory() {
    try {
      await api.delete('/history')
      setHistory([])
    } catch {
      setError('Could not clear history. Check the API connection.')
    }
  }

  const probabilities = prediction?.probabilities ?? {}
  const highest = prediction
    ? Object.entries(probabilities).reduce((best, current) => current[1] > best[1] ? current : best)
    : null

  return (
    <div className="app-shell min-h-screen">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Radiant home">
          <span className="brand-mark"><HeartPulse size={19} strokeWidth={2.3} /></span>
          <span>radiant<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-meta">
          <span className={`service-state ${apiReady ? 'is-ready' : ''}`}>
            <span className="state-dot" />{apiReady ? 'Service ready' : 'Service unavailable'}
          </span>
          <span className="topbar-divider" />
          <span className="workspace-name">IMAGING WORKSPACE</span>
        </div>
        <button className="icon-button top-help" title="About this session" aria-label="About this session" onClick={() => setError('AI-assisted screening only. Results require review by a qualified clinician.')}>
          <ShieldCheck size={18} />
        </button>
      </header>

      <main id="top" className="workspace">
        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> CHEST RADIOGRAPH / ANALYSIS</div>
            <h1>Image review<span className="heading-mark">.</span></h1>
            <p className="page-subtitle">Pneumonia detection and region localization</p>
          </div>
          <div className="session-chip"><Activity size={15} /><span>AI-assisted screening</span></div>
        </section>

        <div className="work-grid">
          <section className="primary-column" aria-label="Image analysis">
            <div className="section-label"><span>01</span><h2>Upload radiograph</h2><span className="section-rule" /></div>
            <div
              className={`upload-zone ${dragging ? 'is-dragging' : ''} ${preview ? 'has-preview' : ''}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]) }}
            >
              <input ref={inputRef} className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.dcm,.dicom,image/jpeg,image/png,application/dicom" onChange={(event) => acceptFile(event.target.files?.[0])} />
              {preview ? (
                <div className="preview-layout">
                  <div className="preview-image-wrap">
                    {file && /\.(dcm|dicom)$/i.test(file.name)
                      ? <div className="dicom-preview"><FileImage size={30} /><span>DICOM STUDY</span><small>Image appears after analysis</small></div>
                      : <img src={preview} alt="Selected chest X-ray preview" />}
                  </div>
                  <div className="preview-details">
                    <span className="file-kicker"><FileImage size={14} /> READY FOR ANALYSIS</span>
                    <strong className="filename" title={file?.name}>{file?.name}</strong>
                    <span className="file-size">{((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB <i /> RGB preview</span>
                    <div className="preview-actions">
                      <button className="text-action" onClick={() => inputRef.current?.click()}><Upload size={15} /> Replace image</button>
                      <button className="icon-button remove-file" title="Remove image" aria-label="Remove image" onClick={() => { setFile(null); setPrediction(null) }}><X size={16} /></button>
                    </div>
                  </div>
                </div>
              ) : (
                <button className="upload-prompt" onClick={() => inputRef.current?.click()}>
                  <span className="upload-icon"><ArrowDownToLine size={22} /></span>
                  <span className="upload-title">Drop an X-ray to begin</span>
                  <span className="upload-caption">or <span className="browse-link">browse files</span> from your device</span>
                  <span className="upload-formats">JPG / JPEG / PNG <i /> UP TO 5 MB</span>
                </button>
              )}
            </div>

            {error && <div className="error-banner" role="alert"><CircleAlert size={17} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></div>}

            <div className="analysis-bar">
              <div className="analysis-note"><ScanLine size={17} /><span>Classification + region localization</span></div>
              <button className="analyze-button" disabled={!file || busy} onClick={() => void runPrediction()}>
                {busy ? <><LoaderCircle className="spin" size={17} /> Analyzing</> : <>Analyze image <ArrowUpRight size={17} /> </>}
              </button>
            </div>

            <div className="section-label results-label"><span>02</span><h2>Image comparison</h2><span className="section-rule" />{prediction && <span className="result-time">{prediction.inference_ms.toFixed(0)} ms</span>}</div>
            <div className="image-pair">
              <ImagePanel title="Original" image={prediction ? `data:image/png;base64,${prediction.original_image}` : preview} emptyText="Original image will appear here" />
              <ImagePanel title="Localized regions" image={prediction ? `data:image/png;base64,${prediction.annotated_image}` : ''} emptyText="Bounding-box view will appear here" annotated />
            </div>
            <p className="image-footnote"><span className="red-key" />Red boxes indicate YOLO-localized regions. Detection does not determine the classification.</p>

            <div className="section-label probability-label"><span>03</span><h2>Class probabilities</h2><span className="section-rule" /></div>
            <div className="probability-panel">
              {classes.map((label, index) => {
                const value = probabilities[label]
                const isHighest = highest?.[0] === label
                return (
                  <div className={`probability-row ${isHighest ? 'is-highest' : ''}`} key={label}>
                    <div className="probability-name"><span className="class-index">0{index + 1}</span><span>{label}</span>{isHighest && <span className="highest-tag"><Check size={12} /> HIGHEST</span>}</div>
                    <div className="probability-track"><span style={{ width: value === undefined ? '0%' : formatPercent(value) }} /></div>
                    <span className="probability-value">{value === undefined ? '--' : formatPercent(value)}</span>
                  </div>
                )
              })}
            </div>
            {prediction && <p className="clinical-note"><ShieldCheck size={15} /> AI-assisted screening only; not a standalone diagnosis.</p>}
          </section>

          <aside className="side-column">
            <section className="finding-panel">
              <div className="panel-topline"><span className="panel-kicker">CURRENT FINDING</span><span className="finding-icon"><Activity size={17} /></span></div>
              {prediction ? (
                <>
                  <p className="finding-label">Classifier output</p>
                  <h2 className="finding-class">{prediction.predicted_class}</h2>
                  <div className="finding-divider" />
                  <div className="metric-row"><span>Class confidence</span><strong>{highest ? formatPercent(highest[1]) : '--'}</strong></div>
                  <div className="metric-row"><span>YOLO max confidence</span><strong>{prediction.yolo_confidence === null ? 'No region' : formatPercent(prediction.yolo_confidence)}</strong></div>
                  {prediction.patient_details && <div className="patient-details">
                    <span className="panel-kicker">PATIENT / STUDY</span>
                    {([
                      ['Patient', prediction.patient_details.patient_name],
                      ['Patient ID', prediction.patient_details.patient_id],
                      ['Age', prediction.patient_details.patient_age],
                      ['Sex', prediction.patient_details.patient_sex],
                      ['Study date', prediction.patient_details.study_date],
                    ] as const).filter(([, value]) => value).map(([label, value]) => (
                      <div className="patient-detail-row" key={label}><span>{label}</span><strong>{value}</strong></div>
                    ))}
                  </div>}
                  <div className="finding-disclaimer"><ShieldCheck size={15} /><span>Classifier is the source of the predicted class. Localization is shown separately.</span></div>
                </>
              ) : (
                <div className="finding-empty"><span className="finding-pulse"><Activity size={21} /></span><strong>Awaiting image</strong><span>Upload a radiograph to see the classifier finding and localization confidence.</span></div>
              )}
            </section>

            <section className="history-panel">
              <div className="history-heading"><div><span className="panel-kicker">THIS SESSION</span><h2>Recent studies <span>{history.length.toString().padStart(2, '0')}</span></h2></div><button className="icon-button clear-button" title="Clear session history" aria-label="Clear session history" disabled={!history.length} onClick={() => void clearHistory()}><Trash2 size={16} /></button></div>
              {history.length ? <ul className="history-list">{history.slice(0, 8).map((item) => <li key={item.id} className="history-item"><span className="history-thumb"><FileImage size={16} /></span><span className="history-info"><strong>{item.filename}</strong><span>{item.predicted_class} <i /> {formatPercent(item.confidence)}</span></span><ChevronRight size={15} className="history-arrow" /></li>)}</ul> : <div className="history-empty"><Clock3 size={17} /><span>No studies in this session</span></div>}
              <div className="history-footer">IN-MEMORY ONLY <span className="history-live-dot" /></div>
            </section>

            <div className="pipeline-note"><div className="pipeline-line"><span className="pipeline-node active" /><span className="pipeline-node" /></div><div><strong>Two-stage workflow</strong><p>ResNet50 + DenseNet121 classify. YOLO localizes.</p></div></div>
          </aside>
        </div>

        <footer className="page-footer"><span>RADIANT IMAGING SYSTEM</span><span>MODEL OUTPUT REQUIRES CLINICAL REVIEW</span><span>V 1.0.0</span></footer>
      </main>
    </div>
  )
}

type ImagePanelProps = { title: string; image: string; emptyText: string; annotated?: boolean }

function ImagePanel({ title, image, emptyText, annotated = false }: ImagePanelProps) {
  return (
    <div className="image-panel">
      <div className="image-panel-head"><span>{title}</span>{annotated && <span className="annotation-legend"><i /> REGIONS</span>}</div>
      <div className={`radiograph-stage ${image ? 'has-image' : ''}`}>
        {image ? <img src={image} alt={annotated ? 'Chest X-ray with red localization boxes' : 'Original chest X-ray'} /> : <div className="image-placeholder"><span className="placeholder-cross"><span /><i /></span><span>{emptyText}</span></div>}
      </div>
    </div>
  )
}

export default App
