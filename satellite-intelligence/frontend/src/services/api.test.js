import { afterEach, describe, expect, it, vi } from 'vitest';
import { clearExpiredSession, getSupabaseSession } from './supabase';
import { acquireCopernicusScene, analyzeImageryScene, api, apiBaseUrl, compareImageryScenes, downloadArtifact, downloadResult, getAnalysisRecord, getAuthStatus, getDataset, getHealth, getImageryAnalysis, getImageryAnalysisTypes, getImageryScene, getResultArtifacts, getResultImage, getResultImageUrl, getResults, getSignedArtifactDownloadUrl, ingestImagery, listImageryScenes, normalizeApiRootUrl, runAllAnalysis, runChangeDetection, runLandcover, runNDBI, runNDVI, runNDWI, searchCopernicusCatalog, uploadBand } from './api';

vi.mock('./supabase', () => ({
  clearExpiredSession: vi.fn().mockResolvedValue(undefined),
  getSupabaseSession: vi.fn().mockResolvedValue(null),
}));

describe('API service', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('calls the backend health and dataset routes', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({
      data: { status: 'ok', service: 'satellite-intelligence-api', provider: 'local', configured: false },
    });
    await expect(getHealth()).resolves.toMatchObject({ status: 'ok', service: 'satellite-intelligence-api' });
    await getDataset();
    expect(get).toHaveBeenNthCalledWith(1, '/health');
    expect(get).toHaveBeenNthCalledWith(2, '/dataset');
  });

  it('normalizes API origins with or without a trailing /api path', () => {
    expect(normalizeApiRootUrl('https://api.example.test/')).toBe('https://api.example.test');
    expect(normalizeApiRootUrl('https://api.example.test/api/')).toBe('https://api.example.test');
    expect(apiBaseUrl).toMatch(/\/api$/);
    expect(apiBaseUrl).not.toMatch(/\/api\/api$/);
  });

  it('rejects an invalid health response instead of marking the API online', async () => {
    vi.spyOn(api, 'get').mockResolvedValue({ data: { status: 'degraded', service: 'satellite-intelligence-api' } });
    await expect(getHealth()).rejects.toThrow('health endpoint returned an unexpected response');
  });

  it('rejects invalid JSON health payloads', async () => {
    vi.spyOn(api, 'get').mockResolvedValue({ data: 'not JSON' });
    await expect(getHealth()).rejects.toThrow('health endpoint returned an unexpected response');
  });

  it('attaches the current Supabase access token to backend requests', async () => {
    getSupabaseSession.mockResolvedValue({ access_token: 'mock-access-token' });
    const interceptor = api.interceptors.request.handlers[0].fulfilled;
    const request = await interceptor({ headers: {} });
    expect(request.headers.Authorization).toBe('Bearer mock-access-token');
  });

  it('adds a stable idempotency key to analysis requests and preserves it on retry', async () => {
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'request-key-1234567890') });
    const interceptor = api.interceptors.request.handlers[0].fulfilled;
    const request = await interceptor({
      method: 'post',
      url: '/analyze/ndvi',
      headers: {},
    });
    const retriedRequest = await interceptor({
      ...request,
      headers: { ...request.headers },
    });

    expect(request.headers['Idempotency-Key']).toBe('request-key-1234567890');
    expect(retriedRequest.headers['Idempotency-Key']).toBe('request-key-1234567890');
  });

  it('checks public auth capability and retrieves protected map images as blobs', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: { authentication_required: true } });
    await expect(getAuthStatus()).resolves.toEqual({ authentication_required: true });
    await getResultImage('result-1', 'ndvi');
    expect(get).toHaveBeenNthCalledWith(1, '/auth/status');
    expect(get).toHaveBeenNthCalledWith(2, '/results/result-1/image/ndvi', { responseType: 'blob' });
  });

  it('preserves HTTP and timeout errors for the UI to report accurately', async () => {
    const get = vi.spyOn(api, 'get');
    get.mockRejectedValueOnce({ response: { status: 503 } });
    await expect(getHealth()).rejects.toMatchObject({ response: { status: 503 } });

    get.mockRejectedValueOnce({ code: 'ECONNABORTED' });
    await expect(getHealth()).rejects.toMatchObject({ code: 'ECONNABORTED' });
  });

  it('clears the local Supabase session and notifies the app after an API 401', async () => {
    const handler = api.interceptors.response.handlers[0].rejected;
    const onExpired = vi.fn();
    window.addEventListener('satellite:auth-expired', onExpired);
    const error = { response: { status: 401 } };

    await expect(handler(error)).rejects.toBe(error);

    expect(clearExpiredSession).toHaveBeenCalledOnce();
    expect(onExpired).toHaveBeenCalledOnce();
    window.removeEventListener('satellite:auth-expired', onExpired);
  });

  it('uses the actual analysis and results endpoints', async () => {
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: {} });
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: {} });
    await runNDVI();
    await runNDWI();
    await runNDBI();
    await runLandcover();
    await runChangeDetection();
    await runAllAnalysis();
    await getResults();
    await getResults(25);
    await getResults(50, { includeIncomplete: true });
    await getAnalysisRecord('analysis-1');
    await downloadResult('id-1', 'ndvi');
    expect(post.mock.calls.map(([path]) => path)).toEqual([
      '/analyze/ndvi',
      '/analyze/ndwi',
      '/analyze/ndbi',
      '/analyze/landcover',
      '/analyze/change',
      '/analyze/all',
    ]);
    expect(get).toHaveBeenLastCalledWith('/results/id-1/download/ndvi', { responseType: 'blob' });
    expect(get.mock.calls).toContainEqual(['/results', { params: { offset: 0 } }]);
    expect(get.mock.calls).toContainEqual(['/results', { params: { offset: 25 } }]);
    expect(get.mock.calls).toContainEqual([
      '/results',
      { params: { offset: 50, include_incomplete: true } },
    ]);
    expect(get.mock.calls).toContainEqual(['/analyses/analysis-1']);
  });

  it('builds a URL for the requested result layer', () => {
    expect(getResultImageUrl('123', 'ndvi')).toContain('/api/results/123/image/ndvi');
  });

  it('lists persisted artifacts and requests an expiring download URL', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: {} });
    await getResultArtifacts('result/1');
    await getSignedArtifactDownloadUrl('result/1', 'summary.json');
    await downloadArtifact('result/1', 'ndvi.tif');
    expect(get).toHaveBeenNthCalledWith(1, '/results/result%2F1/artifacts');
    expect(get).toHaveBeenNthCalledWith(2, '/results/result%2F1/signed-download', {
      params: { artifact_name: 'summary.json' },
    });
    expect(get).toHaveBeenNthCalledWith(
      3,
      '/results/result%2F1/artifacts/ndvi.tif/download',
      { responseType: 'blob' },
    );
  });

  it('submits scene change detection through the authenticated imagery API', async () => {
    const payload = {
      baseline_scene_id: 'a'.repeat(32),
      comparison_scene_id: 'b'.repeat(32),
      baseline_band_mapping: { red: 1, nir: 2 },
      comparison_band_mapping: { red: 1, nir: 2 },
      threshold: 0.1,
    };
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: { status: 'completed' } });

    await expect(compareImageryScenes(payload)).resolves.toEqual({ status: 'completed' });
    expect(post).toHaveBeenCalledWith('/imagery/change-detection', payload, {
      timeout: 300000,
    });
  });

  it('searches the Copernicus STAC catalog through the authenticated API client', async () => {
    const payload = {
      aoi: { type: 'Polygon', coordinates: [] },
      start_date: '2025-01-01',
      end_date: '2025-01-31',
      max_cloud_cover: 20,
      limit: 20,
    };
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: { results: [] } });

    await expect(searchCopernicusCatalog(payload)).resolves.toEqual({ results: [] });
    expect(post).toHaveBeenCalledWith('/satellite/catalog/search', payload, { timeout: 20000 });
  });

  it('uploads GeoTIFF bands to the selected local dataset period', async () => {
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: { success: true } });
    const file = new File(['raster-data'], 'S2_B04.tif', { type: 'image/tiff' });
    await uploadBand('current', file);
    const [path, body, options] = post.mock.calls[0];
    expect(path).toBe('/dataset/current/upload');
    expect(body.get('file')).toBe(file);
    expect(options.onUploadProgress).toBeUndefined();
  });

  it('uploads scenes and retrieves persisted imagery metadata and artifacts', async () => {
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: { success: true } });
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: { scenes: [] } });
    const file = new File(['scene-bytes'], 'scene.tif', { type: 'image/tiff' });
    const onUploadProgress = vi.fn();

    await ingestImagery(file, onUploadProgress);
    await listImageryScenes();
    await getImageryScene('scene-1');
    await downloadArtifact('scene-1', 'preview.png');

    expect(post).toHaveBeenCalledWith(
      '/imagery/ingest',
      expect.any(FormData),
      { onUploadProgress, timeout: 300000 },
    );
    expect(post.mock.calls[0][1].get('file')).toBe(file);
    expect(get).toHaveBeenNthCalledWith(1, '/imagery/scenes');
    expect(get).toHaveBeenNthCalledWith(2, '/imagery/scenes/scene-1');
    expect(get).toHaveBeenNthCalledWith(
      3,
      '/results/scene-1/artifacts/preview.png/download',
      { responseType: 'blob' },
    );
  });

  it('acquires only the selected catalog scene through the backend with an extended timeout', async () => {
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: { id: 'scene-1' } });
    const payload = {
      provider: 'Copernicus Data Space Ecosystem',
      collection: 'sentinel-2-l2a',
      scene_id: 'S2B_MSIL2A_20250131T235509_N0511_R087_T54DWK_20250201T003700',
      aoi: { type: 'Polygon', coordinates: [] },
    };

    await expect(acquireCopernicusScene(payload)).resolves.toEqual({ id: 'scene-1' });

    expect(post).toHaveBeenCalledWith('/satellite/catalog/acquire', payload, {
      timeout: 600000,
    });
  });

  it('uses authenticated scene-analysis endpoints and gives processing an extended timeout', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: {} });
    const post = vi.spyOn(api, 'post').mockResolvedValue({ data: {} });

    await getImageryAnalysisTypes('scene-1');
    await analyzeImageryScene('scene-1', { analysis_type: 'ndvi' });
    await getImageryAnalysis('analysis-1');

    expect(get).toHaveBeenNthCalledWith(1, '/imagery/analysis-types', {
      params: { scene_id: 'scene-1' },
    });
    expect(post).toHaveBeenCalledWith(
      '/imagery/scenes/scene-1/analyses',
      { analysis_type: 'ndvi' },
      { timeout: 300000 },
    );
    expect(get).toHaveBeenNthCalledWith(2, '/imagery/analyses/analysis-1');
  });
});
