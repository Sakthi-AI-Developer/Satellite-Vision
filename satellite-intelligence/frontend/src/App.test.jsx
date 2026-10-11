import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

const service = vi.hoisted(() => ({
  getDataset: vi.fn(),
  getAuthStatus: vi.fn(),
  getHealth: vi.fn(),
  getResults: vi.fn(),
  getSatelliteStatus: vi.fn(),
  getGeoAIStatus: vi.fn(),
  getGeoAIDemoScenario: vi.fn(),
  getGeoAIHistory: vi.fn(),
  runGeoAISpatialAnalysis: vi.fn(),
  runGeoAIVegetationForecast: vi.fn(),
  runGeoAILandCoverTransitions: vi.fn(),
  getGeoAIModelEvaluation: vi.fn(),
  getGeoAIRiskIndicators: vi.fn(),
  runAllAnalysis: vi.fn(),
  runNDVI: vi.fn(),
  runNDWI: vi.fn(),
  runNDBI: vi.fn(),
  runLandcover: vi.fn(),
  runChangeDetection: vi.fn(),
  getResult: vi.fn(),
  getResultArtifacts: vi.fn(),
  uploadBand: vi.fn(),
  listImageryScenes: vi.fn(),
  getImageryScene: vi.fn(),
  ingestImagery: vi.fn(),
  downloadArtifact: vi.fn(),
  getMonitoringStatus: vi.fn(),
  listMonitoringScenes: vi.fn(),
  getMonitoringScene: vi.fn(),
  requestMonitoringScan: vi.fn(),
  downloadMonitoringArtifact: vi.fn(),
}));

const auth = vi.hoisted(() => ({
  getSupabaseSession: vi.fn(),
  signInWithEmail: vi.fn(),
  signUpWithEmail: vi.fn(),
  signOut: vi.fn(),
  subscribeToAuthState: vi.fn(() => () => {}),
}));

vi.mock('./services/api', () => ({
  ...service,
  apiRootUrl: 'http://127.0.0.1:8000',
}));
vi.mock('./services/supabase', () => ({
  ...auth,
  supabaseAuthConfigured: true,
}));
vi.mock('./components/MapView', () => ({
  default: () => <div data-testid="map">Map</div>,
}));

const emptyDataset = {
  current: { available: false, bands: [], errors: [] },
  historical: { available: false, bands: [], errors: [] },
};

beforeEach(() => {
  vi.clearAllMocks();
  service.getAuthStatus.mockResolvedValue({ authentication_required: false });
  auth.getSupabaseSession.mockResolvedValue(null);
  service.getHealth.mockResolvedValue({ status: 'ok', service: 'satellite-intelligence-api' });
  service.getDataset.mockResolvedValue(emptyDataset);
  service.getResults.mockResolvedValue({ results: [] });
  service.getResultArtifacts.mockResolvedValue({ artifacts: [] });
  service.listImageryScenes.mockResolvedValue({ scenes: [] });
  service.getMonitoringStatus.mockResolvedValue({
    enabled: false,
    status: 'disabled',
    disabled_reason: 'Monitoring is disabled by configuration.',
    timezone: 'Asia/Kolkata',
    schedule_time: '03:00',
    scene_counts: {},
    model: { status: 'model_not_ready' },
  });
  service.listMonitoringScenes.mockResolvedValue({ scenes: [] });
  service.requestMonitoringScan.mockResolvedValue({ status: 'accepted' });
  service.downloadMonitoringArtifact.mockResolvedValue(new Blob());
  service.getSatelliteStatus.mockResolvedValue({ configured: false, provider: 'local', message: 'Stage-A local dataset mode is active.' });
  service.getGeoAIStatus.mockResolvedValue({ status: 'ok', available: true, components: ['spatial_analysis', 'forecasting', 'risk'] });
  service.getGeoAIDemoScenario.mockResolvedValue({
    success: true,
    scenario_id: 'synthetic-index-baseline-v1',
    data_classification: 'synthetic',
    acquisition_date: null,
    geographic_coverage: null,
    message: 'This reproducible example is synthetic, not satellite imagery.',
    results: { ndvi: { statistics: { mean: 0.4, valid_pixels: 16 } }, ndwi: { statistics: { mean: 0.1 } }, ndbi: { statistics: { mean: 0.2 } }, landcover: { class_distribution: { 1: {}, 2: {} } } },
  });
  service.getGeoAIHistory.mockResolvedValue({ success: true, status: 'ok', count: 2, mean_ndvi: 0.42, latest_ndvi: 0.44, start_date: '2024-01-01', end_date: '2024-02-01' });
  service.runGeoAISpatialAnalysis.mockResolvedValue({ success: true, status: 'ok', ndvi: { mean: 0.3 }, persistent_change: { mean_ndvi_delta: -0.02 } });
  service.runGeoAIVegetationForecast.mockResolvedValue({ success: true, status: 'ok', forecast_values: [0.4, 0.41], forecast_dates: ['2024-03-01', '2024-03-02'], historical_trend: { slope_per_day: 0.0001 }, prediction_interval: { support: true }, limitations: 'Baseline forecast only.' });
  service.runGeoAILandCoverTransitions.mockResolvedValue({ success: true, status: 'ok', valid_pixel_count: 100, transitions: [{ source_label: 'Vegetation', destination_label: 'Bare ground', pixel_count: 10 }] });
  service.getGeoAIModelEvaluation.mockResolvedValue({ success: true, status: 'ok', rmse: 0.03, mean_absolute_error: 0.02 });
  service.getGeoAIRiskIndicators.mockResolvedValue({ success: true, status: 'ok', count: 1, indicators: [{ severity: 'medium', indicator_name: 'Potential persistent NDVI decline' }] });
  service.uploadBand.mockResolvedValue({ success: true });
  service.runAllAnalysis.mockResolvedValue({ id: 'result-1' });
});

describe('dashboard', () => {
  it('renders real backend status and the empty-dataset instructions', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getAllByText('System Online').length).toBeGreaterThan(0));
    expect(screen.queryByText(/Backend is unavailable|health check failed/)).not.toBeInTheDocument();
    expect(screen.getAllByText('No satellite dataset available.').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/backend\/data\/current\//).length).toBeGreaterThan(0);
    expect(screen.getByText('Georeferenced results will appear here')).toBeInTheDocument();
  });

  it('loads the synthetic demo separately from satellite analysis results', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Demo mode' })).toBeEnabled());

    fireEvent.click(screen.getByRole('button', { name: 'Demo mode' }));

    expect(await screen.findByText('This reproducible example is synthetic, not satellite imagery.')).toBeInTheDocument();
    expect(screen.getByText('NOT SATELLITE DATA')).toBeInTheDocument();
    expect(service.getGeoAIDemoScenario).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Georeferenced results will appear here')).toBeInTheDocument();
  });

  it('does not prefill live-search AOI or dates', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Live Satellite' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Live Satellite' }));

    expect(screen.getByLabelText('AOI GeoJSON')).toHaveValue('');
    expect(screen.getByLabelText('Start date')).toHaveValue('');
    expect(screen.getByLabelText('End date')).toHaveValue('');
  });

  it('shows backend offline state without crashing', async () => {
    service.getHealth.mockRejectedValue(new Error('offline'));
    service.getDataset.mockRejectedValue(new Error('offline'));
    render(<App />);
    expect((await screen.findAllByText('Backend Offline')).length).toBeGreaterThan(0);
    expect(screen.getByText(/Check the configured API URL/)).toBeInTheDocument();
  });

  it('requires sign-in when the backend enables authentication', async () => {
    service.getAuthStatus.mockResolvedValue({ authentication_required: true });
    auth.signInWithEmail.mockResolvedValue({ user: { email: 'analyst@example.test' } });
    render(<App />);

    await screen.findByRole('heading', { name: 'Sign in to Satellite Vision' });
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'analyst@example.test' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'long-test-password' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    await waitFor(() => expect(screen.getByText('analyst@example.test')).toBeInTheDocument());
    expect(auth.signInWithEmail).toHaveBeenCalledWith('analyst@example.test', 'long-test-password');
    expect(service.getResults).toHaveBeenCalled();
  });

  it('restores an existing Supabase session without showing the sign-in form', async () => {
    service.getAuthStatus.mockResolvedValue({ authentication_required: true });
    auth.getSupabaseSession.mockResolvedValue({
      user: { id: 'restored-user', email: 'restored@example.test' },
    });
    render(<App />);

    expect(await screen.findByText('restored@example.test')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Sign in to Satellite Vision' })).not.toBeInTheDocument();
  });

  it('returns to sign-in when the Supabase session is signed out in another tab', async () => {
    let onAuthStateChange;
    auth.subscribeToAuthState.mockImplementation((callback) => {
      onAuthStateChange = callback;
      return () => {};
    });
    service.getAuthStatus.mockResolvedValue({ authentication_required: true });
    auth.getSupabaseSession.mockResolvedValue({
      user: { id: 'active-user', email: 'active@example.test' },
    });
    render(<App />);

    await screen.findByText('active@example.test');
    await waitFor(() => expect(onAuthStateChange).toEqual(expect.any(Function)));
    onAuthStateChange('SIGNED_OUT', null);

    expect(await screen.findByRole('heading', { name: 'Sign in to Satellite Vision' })).toBeInTheDocument();
    expect(screen.queryByText('active@example.test')).not.toBeInTheDocument();
  });

  it('reports HTTP health failures separately from connection failures', async () => {
    service.getHealth.mockRejectedValue({ response: { status: 503 } });
    service.getDataset.mockRejectedValue(new Error('offline'));
    render(<App />);
    expect(await screen.findByText('Backend health check failed with HTTP 503.')).toBeInTheDocument();
  });

  it('reports timed-out API requests instead of presenting an empty result as success', async () => {
    service.getResults.mockRejectedValue({ code: 'ECONNABORTED' });
    const { container } = render(<App />);

    await waitFor(() => expect(container.querySelector('.alert.error')).toHaveTextContent(
      'The request timed out. Check recent results before starting the analysis again.',
    ));
  });

  it('loads older saved results only when requested', async () => {
    service.getResults
      .mockResolvedValueOnce({
        results: [{ id: 'recent-1', analysis: 'NDVI', created_at: '2026-01-01T00:00:00Z' }],
        next_offset: 1,
      })
      .mockResolvedValueOnce({
        results: [{ id: 'older-1', analysis: 'NDWI', created_at: '2025-12-01T00:00:00Z' }],
        next_offset: null,
      });
    render(<App />);

    expect(await screen.findByRole('button', { name: 'Load older results' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Load older results' }));

    await waitFor(() => expect(service.getResults).toHaveBeenLastCalledWith(
      1,
      { includeIncomplete: true },
    ));
    const savedResultSelector = screen.getByLabelText('Load saved result');
    expect(await within(savedResultSelector).findByRole('option', { name: /NDWI/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load older results' })).not.toBeInTheDocument();
  });

  it('disables duplicate complete-analysis requests during loading', async () => {
    service.getDataset.mockResolvedValue({
      current: { available: true, bands: [{ code: 'B04' }, { code: 'B08' }] },
      historical: { available: false, bands: [] },
    });
    let finishRequest;
    service.runAllAnalysis.mockImplementation(() => new Promise((resolve) => { finishRequest = resolve; }));
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: /Run complete analysis/ })).toBeEnabled());
    const runButton = screen.getByRole('button', { name: /Run complete analysis/ });
    fireEvent.click(runButton);
    fireEvent.click(runButton);
    expect(service.runAllAnalysis).toHaveBeenCalledTimes(1);
    expect(await screen.findByRole('button', { name: /Processing imagery/ })).toBeDisabled();
    finishRequest({ id: 'result-1', ndvi: { success: false, message: 'Missing bands' } });
    await waitFor(() => expect(screen.getByText(/Some analyses were unavailable/)).toBeInTheDocument());
  });

  it('loads persisted artifact metadata for a completed analysis', async () => {
    service.getDataset.mockResolvedValue({
      current: { available: true, bands: [{ code: 'B04' }, { code: 'B08' }] },
      historical: { available: false, bands: [] },
    });
    service.runNDVI.mockResolvedValue({
      id: 'result-cloud',
      analysis: 'NDVI',
      result: { statistics: { mean: 0.42 }, bounds: [-1, 50, 0, 51], visualization_url: '/api/results/result-cloud/image/ndvi' },
    });
    service.getResultArtifacts.mockResolvedValue({
      artifacts: [
        { artifact_name: 'ndvi.tif', artifact_type: 'raster', size_bytes: 128 },
        { artifact_name: 'summary.json', artifact_type: 'report', size_bytes: 256 },
      ],
    });
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze NDVI' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Analyze NDVI' }));

    expect(await screen.findByText('ndvi.tif')).toBeInTheDocument();
    expect(screen.getByText('summary.json')).toBeInTheDocument();
    expect(service.getResultArtifacts).toHaveBeenCalledWith('result-cloud');
  });

  it('uploads selected local files to the current dataset and refreshes availability', async () => {
    service.getDataset
      .mockResolvedValueOnce(emptyDataset)
      .mockResolvedValueOnce({
        current: { available: true, bands: [{ code: 'B04' }] },
        historical: { available: false, bands: [] },
      });
    render(<App />);
    await screen.findByLabelText('Upload Current dataset GeoTIFF bands');
    const file = new File(['valid-geotiff-payload'], 'Sentinel_B04.tif', { type: 'image/tiff' });
    fireEvent.change(screen.getByLabelText('Upload Current dataset GeoTIFF bands'), {
      target: { files: [file] },
    });
    await waitFor(() => expect(service.uploadBand).toHaveBeenCalledWith('current', file, expect.any(Function)));
    await waitFor(() => expect(screen.getByText(/1 GeoTIFF band was added to the current dataset/)).toBeInTheDocument());
    expect(service.getDataset).toHaveBeenCalledTimes(2);
  });
});
