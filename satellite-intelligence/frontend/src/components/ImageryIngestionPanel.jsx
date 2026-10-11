import { useCallback, useEffect, useState } from 'react';
import ChangeDetectionPanel from './ChangeDetectionPanel';
import GISWorkspace from './GISWorkspace';
import AIInferencePanel from './AIInferencePanel';
import SceneAnalysisPanel from './SceneAnalysisPanel';
import {
  downloadArtifact,
  getImageryScene,
  ingestImagery,
  listImageryScenes,
} from '../services/api';

function errorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (error.code === 'ECONNABORTED') return 'Imagery upload timed out. Check saved scenes before retrying.';
  return 'Imagery could not be loaded from the backend.';
}

export default function ImageryIngestionPanel({
  savedResults = [],
  resultsLoading = false,
  resultsError = '',
  hasMoreResults = false,
  loadingMoreResults = false,
  onRefreshResults,
  onLoadMoreResults,
  onAnalysisCompleted,
  requestedGisResult,
  requestedGisScene,
}) {
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loadingScenes, setLoadingScenes] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [recentResults, setRecentResults] = useState([]);

  useEffect(() => {
    if (!requestedGisScene?.id) return;
    setScenes((previous) => [
      requestedGisScene,
      ...previous.filter((scene) => scene.id !== requestedGisScene.id),
    ]);
    showScene(requestedGisScene);
  }, [requestedGisScene?.id]);

  const handleAnalysisCompleted = useCallback((result) => {
    setRecentResults((previous) => [
      result,
      ...previous.filter((item) => item.id !== result.id),
    ]);
    onAnalysisCompleted?.(result);
  }, [onAnalysisCompleted]);

  useEffect(() => {
    let active = true;
    listImageryScenes()
      .then((response) => {
        if (active) {
          setScenes((previous) => {
            const listed = response.scenes ?? [];
            if (!requestedGisScene?.id || listed.some((scene) => scene.id === requestedGisScene.id)) {
              return listed;
            }
            return [requestedGisScene, ...listed];
          });
        }
      })
      .catch((requestError) => {
        if (active) setError(errorMessage(requestError));
      })
      .finally(() => {
        if (active) setLoadingScenes(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  async function showScene(scene) {
    setError('');
    setPreviewUrl('');
    try {
      const details = await getImageryScene(scene.id);
      const preview = await downloadArtifact(scene.id, 'preview.png');
      setSelectedScene(details);
      setPreviewUrl(URL.createObjectURL(preview.data));
    } catch (requestError) {
      setSelectedScene(null);
      setError(errorMessage(requestError));
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    setProgress(0);
    setError('');
    try {
      const scene = await ingestImagery(file, (progressEvent) => {
        if (progressEvent.total) {
          setProgress(Math.min(100, Math.round((progressEvent.loaded / progressEvent.total) * 100)));
        }
      });
      setScenes((existing) => [
        scene,
        ...existing.filter((item) => item.id !== scene.id),
      ]);
      await showScene(scene);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setUploading(false);
      setProgress(0);
    }
  }

  async function handleSourceDownload() {
    if (!selectedScene || downloading) return;
    setDownloading(true);
    setError('');
    try {
      const response = await downloadArtifact(selectedScene.id, 'source.tif');
      const objectUrl = URL.createObjectURL(response.data);
      const anchor = document.createElement('a');
      anchor.href = objectUrl;
      anchor.download = metadata.original_filename || 'satellite-scene.tif';
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setDownloading(false);
    }
  }

  const metadata = selectedScene?.metadata;
  return (
    <section className="imagery-ingestion" aria-label="Satellite scene ingestion">
      <div className="panel-header">
        <div>
          <p className="eyebrow">REAL IMAGERY INGESTION</p>
          <h3>Inspect a GeoTIFF scene</h3>
        </div>
      </div>
      <p className="imagery-ingestion-copy">
        Upload a georeferenced GeoTIFF to validate its metadata, create a bounded preview,
        and save the original privately when cloud storage is configured. No satellite
        values or acquisition details are generated when they are absent from the file;
        file metadata and satellite provenance are not independently authenticated.
      </p>
      <label className={`upload-button ${uploading ? 'uploading' : ''}`}>
        <input
          aria-label="Upload satellite scene GeoTIFF"
          type="file"
          accept=".tif,.tiff,image/tiff"
          disabled={uploading}
          onChange={handleUpload}
        />
        {uploading ? `Uploading ${progress}%` : 'Upload scene GeoTIFF'}
      </label>
      {error && <p className="imagery-error" role="alert">{error}</p>}
      <label className="imagery-scene-select">
        <span>Saved scenes</span>
        <select
          aria-label="Saved imagery scenes"
          value={selectedScene?.id ?? ''}
          disabled={loadingScenes || !scenes.length}
          onChange={(event) => {
            const scene = scenes.find((item) => item.id === event.target.value);
            if (scene) showScene(scene);
          }}
        >
          <option value="">
            {loadingScenes ? 'Loading saved scenes…' : scenes.length ? 'Select a saved scene' : 'No saved scenes'}
          </option>
          {scenes.map((scene) => (
            <option key={scene.id} value={scene.id}>
              {scene.metadata?.original_filename ?? scene.id}
            </option>
          ))}
        </select>
      </label>
      <ChangeDetectionPanel
        scenes={scenes}
        loading={loadingScenes}
        onAnalysisCompleted={handleAnalysisCompleted}
      />
      {uploading && <p className="processing-note" role="status">Validating and storing the uploaded scene…</p>}
      {metadata && (
        <div className="imagery-scene-detail" aria-live="polite">
          {previewUrl ? (
            <img className="imagery-preview" src={previewUrl} alt={`Preview of ${metadata.original_filename}`} />
          ) : (
            <p className="processing-note">Loading the private scene preview…</p>
          )}
          <dl>
            <div><dt>Acquisition date</dt><dd>{metadata.acquisition_date ?? 'Not reported in source metadata'}</dd></div>
            <div><dt>CRS</dt><dd>{metadata.crs}</dd></div>
            <div><dt>Raster</dt><dd>{metadata.width} × {metadata.height} · {metadata.band_count} bands</dd></div>
            <div><dt>Bands</dt><dd>{metadata.bands.map((band) => band.code ?? band.name).join(', ')}</dd></div>
            <div><dt>Preview</dt><dd>{metadata.preview.description}</dd></div>
            <div><dt>Persistence</dt><dd>{selectedScene.persistence?.mode === 'cloud-private' ? 'Private cloud storage' : 'Local-only storage'}</dd></div>
          </dl>
          <button
            className="imagery-download"
            type="button"
            disabled={downloading}
            onClick={handleSourceDownload}
          >
            {downloading ? 'Preparing download…' : 'Download original GeoTIFF'}
          </button>
          <SceneAnalysisPanel
            key={selectedScene.id}
            sceneId={selectedScene.id}
            metadata={metadata}
            onAnalysisCompleted={handleAnalysisCompleted}
          />
          <AIInferencePanel
            key={`ai-${selectedScene.id}`}
            sceneId={selectedScene.id}
            onAnalysisCompleted={handleAnalysisCompleted}
          />
        </div>
      )}
      <GISWorkspace
        scenes={scenes}
        scenesLoading={loadingScenes}
        selectedScene={selectedScene}
        recentResults={recentResults}
        savedResults={savedResults}
        resultsLoading={resultsLoading}
        resultsError={resultsError}
        hasMoreResults={hasMoreResults}
        loadingMoreResults={loadingMoreResults}
        onRefreshResults={onRefreshResults}
        onLoadMoreResults={onLoadMoreResults}
        requestedResult={requestedGisResult}
      />
    </section>
  );
}
