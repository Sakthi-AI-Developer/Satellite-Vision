import { useEffect, useState } from 'react';
import {
  downloadArtifact,
  getAIInferenceResult,
  getAIModels,
  runAIInference,
} from '../services/api';

function errorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && typeof detail[0]?.msg === 'string') return detail[0].msg;
  if (error.code === 'ECONNABORTED') {
    return 'AI inference timed out. Check saved results before retrying.';
  }
  return 'AI inference could not be completed. The backend did not confirm a saved result.';
}

export default function AIInferencePanel({ sceneId, onAnalysisCompleted }) {
  const [models, setModels] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState('');
  const [downloading, setDownloading] = useState('');

  useEffect(() => {
    let active = true;
    setLoadingModels(true);
    getAIModels()
      .then((response) => {
        if (active) {
          setModels(response.models ?? []);
          setCandidates(response.candidates ?? []);
        }
      })
      .catch((requestError) => {
        if (active) setError(errorMessage(requestError));
      })
      .finally(() => {
        if (active) setLoadingModels(false);
      });
    return () => {
      active = false;
    };
  }, [sceneId]);

  useEffect(() => {
    if (!analysis?.id || analysis.status !== 'processing') return undefined;
    let active = true;
    let timer;
    const poll = async () => {
      try {
        const response = await getAIInferenceResult(analysis.id);
        if (!active) return;
        setAnalysis(response);
        if (response.status === 'processing') timer = setTimeout(poll, 2000);
      } catch (requestError) {
        if (active) setError(errorMessage(requestError));
      }
    };
    timer = setTimeout(poll, 1500);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [analysis]);

  useEffect(() => {
    if (analysis?.status === 'completed') onAnalysisCompleted?.(analysis);
  }, [analysis, onAnalysisCompleted]);

  async function submitInference() {
    const model = models.find((item) => item.status === 'ready');
    if (!model || submitting) return;
    setSubmitting(true);
    setError('');
    setAnalysis(null);
    try {
      setAnalysis(await runAIInference(sceneId, { model_id: model.id }));
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  }

  async function downloadRaster(artifactName) {
    if (!analysis?.id || downloading) return;
    setDownloading(artifactName);
    setError('');
    try {
      const response = await downloadArtifact(analysis.id, artifactName);
      const objectUrl = URL.createObjectURL(response.data);
      const anchor = document.createElement('a');
      anchor.href = objectUrl;
      anchor.download = artifactName;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setDownloading('');
    }
  }

  const model = models[0];
  const result = analysis?.result;

  return (
    <section className="scene-analysis ai-inference" aria-label="Open-source AI inference">
      <div className="scene-analysis-heading">
        <div>
          <p className="eyebrow">OPTIONAL LOCAL AI · OPEN-SOURCE RUNTIME</p>
          <h3>AI land-cover inference</h3>
        </div>
        {analysis?.status && (
          <span className={`scene-analysis-status status-${analysis.status}`} role="status">
            {analysis.status === 'processing' ? 'Processing AI inference…' : analysis.status}
          </span>
        )}
      </div>
      <p className="scene-analysis-note">
        This is a trained-model inference path, separate from NDVI/NDWI indices and the
        heuristic or K-means baselines. Class names and confidence depend on the locally
        configured model and its training data; confidence is not an accuracy guarantee.
      </p>
      {loadingModels ? (
        <p className="processing-note" role="status">Checking local AI model readiness…</p>
      ) : model?.status === 'ready' ? (
        <div className="ai-model-details">
          <p><strong>{model.name}</strong> · version {model.version} · {model.framework} {model.framework_version}</p>
          <p>License: {model.license}. Input: {model.input_requirements.bands.join(', ')} · {model.input_requirements.normalization}.</p>
          <p>{model.attribution}</p>
        </div>
      ) : (
        <div className="chart-empty" role="status">
          <strong>Model not ready</strong>
          <p>{model?.readiness_detail ?? 'The backend did not report a configured local model.'}</p>
          <p>Configure a trusted model manifest and weights in the backend private model directory; no weights are downloaded automatically.</p>
        </div>
      )}
      {candidates.map((candidate) => (
        <div className="chart-empty" role="status" key={candidate.id}>
          <strong>{candidate.name}: {candidate.status.replaceAll('_', ' ')}</strong>
          <p>{candidate.readiness_detail}</p>
          <p>
            Checkpoint license: {candidate.license}. Expected input:{' '}
            {candidate.input_requirements.bands.join(', ')} ·{' '}
            {candidate.input_requirements.format}.
          </p>
          <a href={candidate.source} target="_blank" rel="noreferrer">
            Review checkpoint and license
          </a>
          <p>No checkpoint is downloaded or executed automatically.</p>
        </div>
      ))}
      {error && <p className="imagery-error" role="alert">{error}</p>}
      {analysis?.error && <p className="imagery-error" role="alert">{analysis.error}</p>}
      <button
        className="secondary-button"
        type="button"
        disabled={loadingModels || submitting || model?.status !== 'ready'}
        onClick={submitInference}
      >
        {submitting ? 'Running local model…' : 'Run AI land-cover inference'}
      </button>
      {analysis?.status === 'processing' && (
        <p className="processing-note" role="status">
          The CPU inference job was accepted. Its status and results are owner-scoped.
        </p>
      )}
      {analysis?.status === 'completed' && result && (
        <div className="ai-inference-result" aria-label="AI inference result" role="region">
          <p>
            Completed with {result.model.name} {result.model.version} in
            {' '}{result.processing_time_seconds} seconds · {result.valid_pixels.toLocaleString()} valid pixels.
          </p>
          <p>
            Mean maximum-class probability: {(result.confidence.mean * 100).toFixed(1)}%
            {' '}({(result.confidence.minimum * 100).toFixed(1)}–{(result.confidence.maximum * 100).toFixed(1)}%).
          </p>
          <p>
            Output grid: {result.rasters.classification.width} × {result.rasters.classification.height}
            {' '}· {result.rasters.classification.crs} · source alignment preserved.
          </p>
          <ul>
            {result.classes.map((item) => (
              <li key={item.class_index}>
                Class {item.class_index}: {String(item.label)} · {item.pixel_count.toLocaleString()} pixels
                {' '}({item.percentage.toFixed(1)}%)
              </li>
            ))}
          </ul>
          {['classification.tif', 'confidence.tif'].map((artifactName) => (
            <button
              key={artifactName}
              className="secondary-button"
              type="button"
              disabled={Boolean(downloading)}
              onClick={() => downloadRaster(artifactName)}
            >
              {downloading === artifactName ? `Downloading ${artifactName}…` : `Download ${artifactName}`}
            </button>
          ))}
          <p className="scene-analysis-note">
            Probability values are model outputs, not calibrated accuracy estimates. Review
            the model license, attribution, training provenance, and validation before use.
          </p>
        </div>
      )}
    </section>
  );
}
