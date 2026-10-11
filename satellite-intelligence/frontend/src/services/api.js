import axios from 'axios';
import { clearExpiredSession, getSupabaseSession } from './supabase';

export function normalizeApiRootUrl(value) {
  return value?.trim().replace(/\/+$/, '').replace(/\/api$/i, '') || '';
}

const configuredApiRootUrl = normalizeApiRootUrl(import.meta.env.VITE_API_URL);
export const apiRootUrl = configuredApiRootUrl || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');
export const apiBaseUrl = `${apiRootUrl}/api`;
export const api = axios.create({ baseURL: apiBaseUrl, timeout: 120000 });

api.interceptors.request.use(async (config) => {
  const session = await getSupabaseSession();
  if (session?.access_token) {
    config.headers.Authorization = `Bearer ${session.access_token}`;
  }
  if (
    config.method === 'post'
    && /^\/analyze(?:\/|$)/.test(config.url ?? '')
    && !config.headers['Idempotency-Key']
  ) {
    const idempotencyKey = globalThis.crypto?.randomUUID?.();
    if (idempotencyKey) config.headers['Idempotency-Key'] = idempotencyKey;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401 && typeof window !== 'undefined') {
      let sessionClearFailed = false;
      try {
        await clearExpiredSession();
      } catch {
        sessionClearFailed = true;
      }
      window.dispatchEvent(new CustomEvent('satellite:auth-expired', {
        detail: { sessionClearFailed },
      }));
    }
    return Promise.reject(error);
  },
);

export async function getHealth() {
  if (!apiRootUrl) {
    throw new Error('VITE_API_URL is required for production builds.');
  }
  const { data } = await api.get('/health');
  if (data?.status !== 'ok' || data?.service !== 'satellite-intelligence-api') {
    throw new Error('The API health endpoint returned an unexpected response.');
  }
  return data;
}

export async function getAuthStatus() {
  return (await api.get('/auth/status')).data;
}

export async function getDataset() {
  return (await api.get('/dataset')).data;
}

export async function uploadBand(period, file, onUploadProgress) {
  const body = new FormData();
  body.append('file', file);
  return (await api.post(`/dataset/${encodeURIComponent(period)}/upload`, body, {
    onUploadProgress,
  })).data;
}

export async function ingestImagery(file, onUploadProgress) {
  const body = new FormData();
  body.append('file', file);
  return (await api.post('/imagery/ingest', body, {
    onUploadProgress,
    timeout: 300000,
  })).data;
}

export async function listImageryScenes() {
  return (await api.get('/imagery/scenes')).data;
}

export async function getImageryScene(sceneId) {
  return (await api.get(`/imagery/scenes/${encodeURIComponent(sceneId)}`)).data;
}

export async function getImageryAnalysisTypes(sceneId) {
  return (await api.get('/imagery/analysis-types', {
    params: sceneId ? { scene_id: sceneId } : undefined,
  })).data;
}

export async function analyzeImageryScene(sceneId, payload) {
  return (await api.post(
    `/imagery/scenes/${encodeURIComponent(sceneId)}/analyses`,
    payload,
    { timeout: 300000 },
  )).data;
}

export async function getImageryAnalysis(analysisId) {
  return (await api.get(`/imagery/analyses/${encodeURIComponent(analysisId)}`)).data;
}

export async function getAIModels() {
  return (await api.get('/imagery/ai-models')).data;
}

export async function getMonitoringStatus() {
  return (await api.get('/monitoring/status')).data;
}

export async function listMonitoringScenes(params = {}) {
  return (await api.get('/monitoring/scenes', { params })).data;
}

export async function getMonitoringScene(sceneId) {
  return (await api.get(`/monitoring/scenes/${encodeURIComponent(sceneId)}`)).data;
}

export async function requestMonitoringScan() {
  return (await api.post('/monitoring/scan')).data;
}

export async function downloadMonitoringArtifact(sceneId, artifactName) {
  return (await api.get(
    `/monitoring/scenes/${encodeURIComponent(sceneId)}/artifacts/${encodeURIComponent(artifactName)}`,
    { responseType: 'blob' },
  )).data;
}

export async function runAIInference(sceneId, payload) {
  return (await api.post(
    `/imagery/scenes/${encodeURIComponent(sceneId)}/ai-inference`,
    payload,
    { timeout: 300000 },
  )).data;
}

export async function getAIInferenceResult(analysisId) {
  return (await api.get(`/imagery/ai-inference/${encodeURIComponent(analysisId)}`)).data;
}

export async function compareImageryScenes(payload) {
  return (await api.post('/imagery/change-detection', payload, {
    timeout: 300000,
  })).data;
}

export async function runNDVI() {
  return (await api.post('/analyze/ndvi')).data;
}

export async function runNDWI() {
  return (await api.post('/analyze/ndwi')).data;
}

export async function runNDBI() {
  return (await api.post('/analyze/ndbi')).data;
}

export async function runLandcover() {
  return (await api.post('/analyze/landcover')).data;
}

export async function runChangeDetection() {
  return (await api.post('/analyze/change')).data;
}

export async function runAllAnalysis() {
  return (await api.post('/analyze/all')).data;
}

export async function getSatelliteStatus() {
  return (await api.get('/satellite/status')).data;
}

export async function searchSatellite(payload) {
  return (await api.post('/satellite/search', payload)).data;
}

export async function searchCopernicusCatalog(payload) {
  return (await api.post('/satellite/catalog/search', payload, { timeout: 20000 })).data;
}

export async function acquireCopernicusScene(payload) {
  return (await api.post('/satellite/catalog/acquire', payload, {
    timeout: 600000,
  })).data;
}

export async function downloadSatellite(payload) {
  return (await api.post('/satellite/download', payload)).data;
}

export async function listSatelliteProducts() {
  return (await api.get('/satellite/products')).data;
}

export async function getResults(offset = 0, { includeIncomplete = false } = {}) {
  return (await api.get('/results', {
    params: { offset, ...(includeIncomplete ? { include_incomplete: true } : {}) },
  })).data;
}

export async function getMetadata() {
  return (await api.get('/metadata')).data;
}

export async function getDataQuality() {
  return (await api.get('/data-quality')).data;
}

export async function getModelStatus() {
  return (await api.get('/model/status')).data;
}

export async function getGeoAIStatus() {
  return (await api.get('/geoai/status')).data;
}

export async function getGeoAIDemoScenario() {
  return (await api.get('/geoai/demo')).data;
}

export async function runGeoAISpatialAnalysis(payload) {
  return (await api.post('/geoai/spatial-analysis', payload)).data;
}

export async function runGeoAIVegetationForecast(payload) {
  return (await api.post('/geoai/vegetation-forecast', payload)).data;
}

export async function runGeoAILandCoverTransitions(payload) {
  return (await api.post('/geoai/land-cover-transitions', payload)).data;
}

export async function getGeoAIModelEvaluation() {
  return (await api.get('/geoai/model-evaluation')).data;
}

export async function getGeoAIRiskIndicators() {
  return (await api.get('/geoai/risk-indicators')).data;
}

export async function getGeoAIHistory() {
  return (await api.get('/geoai/history')).data;
}

export async function getResult(resultId) {
  return (await api.get(`/results/${encodeURIComponent(resultId)}`)).data;
}

export async function getAnalysisRecord(analysisId) {
  return (await api.get(`/analyses/${encodeURIComponent(analysisId)}`)).data;
}

export async function getResultArtifacts(resultId) {
  return (await api.get(`/results/${encodeURIComponent(resultId)}/artifacts`)).data;
}

export async function getSignedArtifactDownloadUrl(resultId, artifactName) {
  return (await api.get(`/results/${encodeURIComponent(resultId)}/signed-download`, {
    params: { artifact_name: artifactName },
  })).data;
}

export async function getVisualizationUrl(kind) {
  return `${apiBaseUrl}/visualization/${encodeURIComponent(kind)}`;
}

export async function downloadResult(resultId, fileKey) {
  return api.get(
    `/results/${encodeURIComponent(resultId)}/download/${encodeURIComponent(fileKey)}`,
    { responseType: 'blob' },
  );
}

export async function downloadArtifact(resultId, artifactName) {
  return api.get(
    `/results/${encodeURIComponent(resultId)}/artifacts/${encodeURIComponent(artifactName)}/download`,
    { responseType: 'blob' },
  );
}

export async function getResultImage(resultId, analysisKey) {
  return api.get(
    `/results/${encodeURIComponent(resultId)}/image/${encodeURIComponent(analysisKey)}`,
    { responseType: 'blob' },
  );
}

export function getResultImageUrl(resultId, analysisKey) {
  return `${apiBaseUrl}/results/${encodeURIComponent(resultId)}/image/${encodeURIComponent(analysisKey)}`;
}

export function getApiResourceUrl(path) {
  return new URL(path, `${apiRootUrl}/`).toString();
}
