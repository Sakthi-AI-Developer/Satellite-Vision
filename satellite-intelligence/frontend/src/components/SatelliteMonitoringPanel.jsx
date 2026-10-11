import { useCallback, useEffect, useState } from 'react';
import {
  downloadMonitoringArtifact,
  getMonitoringStatus,
  listMonitoringScenes,
  requestMonitoringScan,
} from '../services/api';

function formatDate(value) {
  if (!value) return 'Not recorded';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? 'Not recorded'
    : date.toLocaleString(undefined, { timeZone: 'UTC', timeZoneName: 'short' });
}

export default function SatelliteMonitoringPanel() {
  const [status, setStatus] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, history] = await Promise.all([
        getMonitoringStatus(),
        listMonitoringScenes({ limit: 10 }),
      ]);
      setStatus(nextStatus);
      setScenes(history.scenes ?? []);
      setError('');
    } catch (requestError) {
      setError(requestError.response?.data?.detail ?? 'Monitoring status could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 60_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const handleManualScan = async () => {
    setTriggering(true);
    setError('');
    try {
      await requestMonitoringScan();
      await refresh();
    } catch (requestError) {
      setError(requestError.response?.data?.detail ?? 'The scan could not be started.');
    } finally {
      setTriggering(false);
    }
  };

  const handleDownload = async (sceneId, artifactName) => {
    try {
      const blob = await downloadMonitoringArtifact(sceneId, artifactName);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${sceneId}-${artifactName}`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (requestError) {
      setError(requestError.response?.data?.detail ?? 'The analysis artifact could not be downloaded.');
    }
  };

  const counts = status?.scene_counts ?? {};
  return (
    <section className="panel" aria-label="Daily satellite monitoring">
      <div className="panel-header">
        <div>
          <p className="eyebrow">AUTOMATED OBSERVATIONS</p>
          <h2>Daily satellite monitoring</h2>
        </div>
        <span className="stage-badge">
          {status?.enabled ? `MONITOR ${status.status?.toUpperCase() ?? 'READY'}` : 'MONITOR DISABLED'}
        </span>
      </div>
      {loading && <p role="status">Loading monitoring status…</p>}
      {error && <p className="alert error" role="alert">{error}</p>}
      {!loading && status && (
        <>
          {!status.enabled && <p className="panel-note">{status.disabled_reason}</p>}
          <div className="ndvi-stat-grid">
            <div className="ndvi-stat"><small>Last successful catalog scan</small><b>{formatDate(status.last_successful_scan_at)}</b></div>
            <div className="ndvi-stat"><small>Next scheduled run</small><b>{formatDate(status.next_scheduled_run_at)}</b></div>
            <div className="ndvi-stat"><small>Schedule timezone</small><b>{status.timezone} · {status.schedule_time}</b></div>
            <div className="ndvi-stat"><small>Latest observation</small><b>{formatDate(status.latest_observation_at)}</b></div>
            <div className="ndvi-stat">
              <small>Local model readiness</small>
              <b>{status.model?.status === 'ready'
                ? `${status.model.name} · ${status.model.version ?? 'version unknown'}`
                : status.model?.status ?? 'unknown'}</b>
            </div>
          </div>
          <div className="ndvi-stat-grid" style={{ marginTop: 10 }}>
            <div className="ndvi-stat"><small>Discovered</small><b>{counts.discovered ?? 0}</b></div>
            <div className="ndvi-stat"><small>Downloaded and validated</small><b>{counts.downloaded ?? 0}</b></div>
            <div className="ndvi-stat"><small>Analyzed</small><b>{counts.analyzed ?? 0}</b></div>
            <div className="ndvi-stat"><small>Rejected / failed</small><b>{counts.rejected ?? 0} / {counts.failed ?? 0}</b></div>
          </div>
          <p className="panel-note">
            Source: {status.data_source} · {status.collection}. Observation, catalog availability, and processing timestamps are shown separately below.
          </p>
          {status.last_error && <p className="alert error" role="alert">{status.last_error}</p>}
          <button
            type="button"
            className="secondary-button"
            onClick={handleManualScan}
            disabled={!status.enabled || status.status === 'running' || triggering}
          >
            {triggering ? 'Starting scan…' : 'Scan now'}
          </button>
          <div style={{ display: 'grid', gap: 8, marginTop: 14 }}>
            <h3 style={{ margin: 0 }}>Recent scenes</h3>
            {scenes.length === 0
              ? <p className="panel-note">No satellite scenes have been recorded.</p>
              : scenes.map((scene) => (
                <article key={scene.scene_id} className="panel" aria-label={`Scene ${scene.scene_id}`}>
                  <b>{scene.scene_id}</b>
                  <p>Status: {scene.status} · Cloud cover: {scene.cloud_cover ?? 'unknown'}%</p>
                  <p>Observation time: {formatDate(scene.observation_at)}</p>
                  <p>Catalog availability time: {formatDate(scene.published_at)}</p>
                  <p>First discovered: {formatDate(scene.discovered_at)}</p>
                  <p>Last processing: {formatDate(scene.processing_completed_at)}</p>
                  <p>Geographic bounds: {scene.bbox?.join(', ') ?? 'Not reported by catalog'}</p>
                  <p>Attempts: {scene.attempt_count} · Next retry: {formatDate(scene.next_retry_at)}</p>
                  {scene.last_error && <p role="alert">{scene.last_error}</p>}
                  {scene.classification_url && (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => handleDownload(scene.scene_id, 'classification.tif')}
                    >
                      Download real classification raster
                    </button>
                  )}
                  {scene.confidence_url && (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => handleDownload(scene.scene_id, 'confidence.tif')}
                    >
                      Download confidence raster
                    </button>
                  )}
                </article>
              ))}
          </div>
        </>
      )}
    </section>
  );
}
