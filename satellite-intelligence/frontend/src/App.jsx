import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiRootUrl, downloadSatellite, getAuthStatus, getDataset, getGeoAIDemoScenario, getGeoAIHistory, getGeoAIModelEvaluation, getGeoAIRiskIndicators, getGeoAIStatus, getHealth, getResults, getResult, getResultArtifacts, getSatelliteStatus, runAllAnalysis, runChangeDetection, runGeoAILandCoverTransitions, runGeoAISpatialAnalysis, runGeoAIVegetationForecast, runLandcover, runNDBI, runNDVI, runNDWI, searchSatellite, uploadBand } from './services/api';
import AnalysisHistory from './components/AnalysisHistory';
import AnalysisPanel from './components/AnalysisPanel';
import AuthPanel from './components/AuthPanel';
import DatasetSelector from './components/DatasetSelector';
import DownloadPanel from './components/DownloadPanel';
import ErrorMessage from './components/ErrorMessage';
import Header from './components/Header';
import InsightPanel from './components/InsightPanel';
import Legend from './components/Legend';
import LoadingIndicator from './components/LoadingIndicator';
import MetricCard from './components/MetricCard';
import SatelliteCatalogSearch from './components/SatelliteCatalogSearch';
import SatelliteMonitoringPanel from './components/SatelliteMonitoringPanel';
import {
  getSupabaseSession,
  signOut,
  subscribeToAuthState,
  supabaseAuthConfigured,
} from './services/supabase';

const MapView = lazy(() => import('./components/MapView'));
const LandCoverChart = lazy(() => import('./components/LandCoverChart'));
const ChangeChart = lazy(() => import('./components/ChangeChart'));

const layerConfig = [
  { key: 'ndvi', label: 'NDVI', color: '#83bf79' },
  { key: 'ndwi', label: 'NDWI', color: '#5da9e9' },
  { key: 'ndbi', label: 'NDBI', color: '#dc846e' },
  { key: 'landcover', label: 'Land Use', color: '#d0ad67' },
  { key: 'change_detection', label: 'Change Detection', color: '#ae7fd0' },
];
const analysisActions = {
  ndvi: runNDVI,
  ndwi: runNDWI,
  ndbi: runNDBI,
  landcover: runLandcover,
  change_detection: runChangeDetection,
};
const number = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '--';
const percent = (value) => Number.isFinite(value) ? `${value.toFixed(1)}%` : '--';

function analysisMapFromResponse(response) {
  if (response?.result) return {};
  return Object.fromEntries(
    layerConfig
      .filter(({ key }) => response?.[key]?.success)
      .map(({ key }) => [key, response[key]]),
  );
}

function friendlyError(error) {
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return 'The request timed out. Check recent results before starting the analysis again.';
  }
  if (!error.response) return 'Backend is unavailable. Check the API connection; results were not updated.';
  if (error.response.status === 401) return 'Your sign-in expired. Sign in again to continue.';
  const detail = error.response.data?.detail;
  return typeof detail === 'string' ? detail : 'Unable to process the selected dataset.';
}

function logSafeRequestFailure(event, error) {
  console.error(event, {
    status: error?.response?.status,
    code: typeof error?.code === 'string' ? error.code : undefined,
  });
}

function healthErrorMessage(error) {
  if (error.response) return `Backend health check failed with HTTP ${error.response.status}.`;
  if (error.code === 'ECONNABORTED') return 'Backend health check timed out. Check the API URL and try again.';
  if (!apiRootUrl) return 'Production API URL is not configured. Set VITE_API_URL and rebuild the frontend.';
  if (error.message?.includes('unexpected response')) return 'Backend health check returned an invalid response.';
  return 'Backend is unavailable. Check the configured API URL and network connection.';
}

function App() {
  const [authMode, setAuthMode] = useState('checking');
  const [authRequired, setAuthRequired] = useState(false);
  const [authUser, setAuthUser] = useState(null);
  const [authNotice, setAuthNotice] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [backendError, setBackendError] = useState('');
  const [dataset, setDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [uploadingPeriod, setUploadingPeriod] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analyses, setAnalyses] = useState({});
  const [resultId, setResultId] = useState(null);
  const [recentResults, setRecentResults] = useState([]);
  const [gisRequest, setGisRequest] = useState(null);
  const [nextResultsOffset, setNextResultsOffset] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(true);
  const [resultsError, setResultsError] = useState('');
  const [loadingOlderResults, setLoadingOlderResults] = useState(false);
  const [resultLoading, setResultLoading] = useState(false);
  const [resultArtifacts, setResultArtifacts] = useState([]);
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [artifactsError, setArtifactsError] = useState('');
  const [activeLayer, setActiveLayer] = useState('ndvi');
  const [dataSource, setDataSource] = useState('local');
  const [satelliteStatus, setSatelliteStatus] = useState({ configured: false, provider: 'local', message: 'Stage-A local dataset mode is active.' });
  const [geoAIStatus, setGeoAIStatus] = useState({ status: 'checking', available: false, components: [] });
  const [geoAIInsights, setGeoAIInsights] = useState({
    history: null,
    forecast: null,
    transitions: null,
    spatial: null,
    risk: null,
    evaluation: null,
  });
  const [aoi, setAoi] = useState(null);
  const [satelliteResults, setSatelliteResults] = useState([]);
  const [selectedProductId, setSelectedProductId] = useState('');
  const [selectedCatalogScene, setSelectedCatalogScene] = useState(null);
  const [requestedGisScene, setRequestedGisScene] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [demoSummary, setDemoSummary] = useState(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const analysisRequestActive = useRef(false);

  const clearPrivateWorkspace = useCallback(() => {
    setDataset(null);
    setAnalyses({});
    setResultId(null);
    setRecentResults([]);
    setGisRequest(null);
    setResultsError('');
    setResultArtifacts([]);
    setArtifactsError('');
    setDemoSummary(null);
    setIsDemoMode(false);
    setSatelliteResults([]);
    setSelectedProductId('');
    setSelectedCatalogScene(null);
    setRequestedGisScene(null);
    setGeoAIInsights({
      history: null,
      forecast: null,
      transitions: null,
      spatial: null,
      risk: null,
      evaluation: null,
    });
  }, []);

  const refreshDataset = useCallback(async () => {
    setDatasetLoading(true);
    try {
      setDataset(await getDataset());
    } catch (requestError) {
      setError(friendlyError(requestError));
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  const refreshResults = useCallback(async () => {
    setResultsLoading(true);
    setResultsError('');
    try {
      const response = await getResults(0, { includeIncomplete: true });
      setRecentResults(response.results ?? []);
      setNextResultsOffset(response.next_offset ?? null);
    } catch (requestError) {
      setRecentResults([]);
      setNextResultsOffset(null);
      const message = friendlyError(requestError);
      setResultsError(message);
      setError(message);
    } finally {
      setResultsLoading(false);
    }
  }, []);

  const loadOlderResults = useCallback(async () => {
    if (nextResultsOffset === null || loadingOlderResults) return;
    setLoadingOlderResults(true);
    try {
      const response = await getResults(nextResultsOffset, { includeIncomplete: true });
      setRecentResults((previous) => {
        const existingIds = new Set(previous.map((item) => item.id));
        return [...previous, ...(response.results ?? []).filter((item) => !existingIds.has(item.id))];
      });
      setNextResultsOffset(response.next_offset ?? null);
      setResultsError('');
    } catch (requestError) {
      const message = friendlyError(requestError);
      setResultsError(message);
      setError(message);
    } finally {
      setLoadingOlderResults(false);
    }
  }, [loadingOlderResults, nextResultsOffset]);

  const loadResultArtifacts = useCallback(async (id) => {
    setArtifactsLoading(true);
    setArtifactsError('');
    try {
      const response = await getResultArtifacts(id);
      setResultArtifacts(response.artifacts ?? []);
    } catch {
      setResultArtifacts([]);
      setArtifactsError('Saved artifact metadata could not be loaded. Try selecting this result again.');
    } finally {
      setArtifactsLoading(false);
    }
  }, []);

  const loadGeoAIInsights = useCallback(async () => {
    try {
      const status = await getGeoAIStatus();
      setGeoAIStatus(status);
    } catch (error) {
      logSafeRequestFailure('Failed to fetch GeoAI status.', error);
      setGeoAIStatus({ status: 'unavailable', available: false, components: [] });
    }

    try {
      const [history, forecast, transitions, spatial, risk, evaluation] = await Promise.all([
        getGeoAIHistory().catch(() => ({ success: false, status: 'insufficient-data', message: 'Historic observations are unavailable.' })),
        runGeoAIVegetationForecast({ horizon_days: 90 }).catch(() => ({ success: false, status: 'insufficient-data', message: 'Vegetation forecast is unavailable.' })),
        runGeoAILandCoverTransitions({ current_dataset_id: 'current', historical_dataset_id: 'historical', include_area: true }).catch(() => ({ success: false, status: 'insufficient-data', message: 'Land-cover transitions are unavailable.' })),
        runGeoAISpatialAnalysis({ dataset_id: 'current', include_history: true, block_size: 10 }).catch(() => ({ success: false, status: 'insufficient-data', message: 'Spatial analysis is unavailable.' })),
        getGeoAIRiskIndicators().catch(() => ({ success: false, status: 'insufficient-data', message: 'Risk indicators are unavailable.' })),
        getGeoAIModelEvaluation().catch(() => ({ success: false, status: 'insufficient-data', message: 'Model evaluation is unavailable.' })),
      ]);
      setGeoAIInsights({ history, forecast, transitions, spatial, risk, evaluation });
    } catch (error) {
      logSafeRequestFailure('Failed to refresh GeoAI insights.', error);
      setGeoAIInsights({ history: null, forecast: null, transitions: null, spatial: null, risk: null, evaluation: null });
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    async function checkAuthentication() {
      try {
        const [status, session] = await Promise.all([getAuthStatus(), getSupabaseSession()]);
        if (!mounted) return;
        const required = Boolean(status.authentication_required);
        setAuthRequired(required);
        if (required && !supabaseAuthConfigured) {
          setAuthMode('configuration-error');
          return;
        }
        if (required && !session) {
          setAuthMode('login');
          return;
        }
        setAuthUser(required ? session?.user ?? null : null);
        setAuthMode('ready');
      } catch (error) {
        if (mounted) {
          setBackendStatus('offline');
          setBackendError(healthErrorMessage(error));
          setAuthMode('unavailable');
        }
      }
    }
    checkAuthentication();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (authMode === 'checking' || !authRequired || !supabaseAuthConfigured) {
      return undefined;
    }
    return subscribeToAuthState((event, session) => {
      if (event === 'SIGNED_OUT') {
        clearPrivateWorkspace();
        setAuthUser(null);
        setAuthMode(authRequired ? 'login' : 'ready');
        return;
      }
      if (session?.user) {
        if (authUser?.id !== session.user.id) clearPrivateWorkspace();
        setAuthUser(session.user);
        if (authRequired) setAuthMode('ready');
      }
    });
  }, [authMode, authRequired, authUser?.id, clearPrivateWorkspace]);

  useEffect(() => {
    function handleExpiredSession(event) {
      clearPrivateWorkspace();
      setAuthUser(null);
      setAuthNotice(
        event.detail?.sessionClearFailed
          ? 'Your session expired. Sign in again; this browser could not clear the expired session automatically.'
          : 'Your session expired. Sign in again to continue.',
      );
      setAuthMode(authRequired ? 'login' : 'ready');
    }
    window.addEventListener('satellite:auth-expired', handleExpiredSession);
    return () => window.removeEventListener('satellite:auth-expired', handleExpiredSession);
  }, [authRequired, clearPrivateWorkspace]);

  useEffect(() => {
    if (authMode !== 'ready') return undefined;
    let mounted = true;
    async function initialize() {
      const healthCheck = getHealth()
        .then(() => { if (mounted) setBackendStatus('online'); })
        .catch((error) => {
          if (mounted) {
            setBackendStatus('offline');
            setBackendError(healthErrorMessage(error));
          }
        });
      const datasetCheck = refreshDataset();
      const resultsCheck = refreshResults();
      const satelliteCheck = getSatelliteStatus()
        .then((response) => { if (mounted) setSatelliteStatus(response); })
        .catch(() => { if (mounted) setSatelliteStatus({ configured: false, provider: 'local', message: 'Live satellite provider is not configured.' }); });
      const geoAIStatusCheck = loadGeoAIInsights();
      await Promise.all([healthCheck, datasetCheck, resultsCheck, satelliteCheck, geoAIStatusCheck]);
    }
    initialize();
    return () => { mounted = false; };
  }, [authMode, loadGeoAIInsights, refreshDataset, refreshResults]);

  const current = analyses.ndvi;
  const ndwi = analyses.ndwi;
  const ndbi = analyses.ndbi;
  const landcover = analyses.landcover;
  const change = analyses.change_detection;
  const selected = analyses[activeLayer];
  const mapLayer = layerConfig.find((layer) => layer.key === activeLayer);
  const selectedProvenance = selected?.provenance?.datasets ?? {};

  const insights = useMemo(() => {
    const messages = [];
    if (change?.mean_change < 0) messages.push('Satellite-derived vegetation index is lower than the historical reference period.');
    if (change?.mean_change > 0) messages.push('Satellite-derived vegetation index is higher than the historical reference period.');
    if (current?.vegetation_percentage >= 50) messages.push('Based on available imagery, a relatively high share of valid pixels is above the configured vegetation-index threshold.');
    if (ndbi?.built_up_indicator_percentage > 50) messages.push('The satellite-derived built-up indicator is positive for a relatively large share of valid pixels; this is not definitive building detection.');
    if (!messages.length && !current) messages.push('Run an analysis to generate satellite-derived indicators and summaries.');
    if (!messages.length && current) messages.push('Indicators are calculated from available imagery; the results require ground validation where applicable.');
    return messages;
  }, [change, current, ndbi]);

  const acceptAllResponse = (response) => {
    const successful = Object.fromEntries(
      Object.entries(analysisMapFromResponse(response))
        .map(([key, value]) => [key, { ...value, result_id: response.id }]),
    );
    setAnalyses((previous) => ({ ...previous, ...successful }));
    const unavailable = layerConfig
      .filter(({ key }) => response[key] && !response[key].success)
      .map(({ label, key }) => `${label}: ${response[key].message}`);
    setNotice(unavailable.length ? `Some analyses were unavailable. ${unavailable.join(' ')}` : 'Analysis complete. Results are calculated from the available local rasters.');
  };

  async function handleAnalysis(key) {
    if (analysisRequestActive.current) return;
    analysisRequestActive.current = true;
    setBusy(key);
    setError('');
    setNotice('');
    try {
      const response = await analysisActions[key]();
      setAnalyses((previous) => ({
        ...previous,
        [key]: { ...response.result, result_id: response.id },
      }));
      setResultId(response.id);
      await loadResultArtifacts(response.id);
      setNotice(`${response.analysis} completed using local satellite imagery.`);
      await refreshResults();
    } catch (requestError) {
      setError(friendlyError(requestError));
    } finally {
      analysisRequestActive.current = false;
      setBusy('');
    }
  }

  async function handleAllAnalysis() {
    if (analysisRequestActive.current) return;
    analysisRequestActive.current = true;
    setBusy('all');
    setError('');
    setNotice('');
    try {
      const response = await runAllAnalysis();
      setResultId(response.id);
      acceptAllResponse(response);
      await loadResultArtifacts(response.id);
      await refreshResults();
    } catch (requestError) {
      setError(friendlyError(requestError));
    } finally {
      analysisRequestActive.current = false;
      setBusy('');
    }
  }

  async function handleUpload(period, files) {
    if (uploadingPeriod) return;
    setUploadingPeriod(period);
    setUploadProgress(0);
    setError('');
    setNotice('');
    let uploaded = 0;
    const failures = [];
    try {
      for (const [index, file] of files.entries()) {
        try {
          await uploadBand(period, file, (event) => {
            if (event.total) {
              const fileProgress = Math.round((event.loaded / event.total) * 100);
              setUploadProgress(Math.round(((index + fileProgress / 100) / files.length) * 100));
            }
          });
          uploaded += 1;
        } catch (uploadError) {
          const detail = uploadError.response?.data?.detail;
          failures.push(`${file.name}: ${typeof detail === 'string' ? detail : 'Upload failed.'}`);
        }
      }
      await refreshDataset();
      if (uploaded) {
        setNotice(`${uploaded} GeoTIFF ${uploaded === 1 ? 'band was' : 'bands were'} added to the ${period} dataset.`);
      }
      if (failures.length) setError(failures.join(' '));
    } finally {
      setUploadingPeriod('');
      setUploadProgress(0);
    }
  }

  async function handleSelectResult(event) {
    const id = event.target.value;
    if (!id) return;
    setResultLoading(true);
    setError('');
    try {
      const response = await getResult(id);
      setResultId(id);
      await loadResultArtifacts(id);
      if (response.result) {
        const analysisTitle = response.result.analysis?.toLowerCase() ?? '';
        const matching = layerConfig.find(({ key, label }) =>
          analysisTitle.includes(label.toLowerCase())
          || (key === 'landcover' && analysisTitle.includes('land-cover'))
          || (key === 'change_detection' && analysisTitle.includes('change')),
        );
        if (matching) {
          setAnalyses((previous) => ({
            ...previous,
            [matching.key]: { ...response.result, result_id: id },
          }));
        }
      } else {
        const loaded = Object.fromEntries(
          Object.entries(analysisMapFromResponse(response))
            .map(([key, value]) => [key, { ...value, result_id: id }]),
        );
        setAnalyses((previous) => ({ ...previous, ...loaded }));
      }
    } catch (requestError) {
      setError(friendlyError(requestError));
    } finally {
      setResultLoading(false);
    }
  }

  async function handleSatelliteSearch(criteria) {
    setAoi(criteria.aoi);
    setLiveLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await searchSatellite(criteria);
      if (!response.results?.length) {
        setSatelliteResults([]);
        setNotice('No scenes were found for the selected AOI and date range.');
        return;
      }
      setSatelliteResults(response.results);
      setSelectedProductId(response.results[0].product_id);
      setNotice(response.provider === 'mock'
        ? 'A synthetic fixture was returned for demo use. It is not a satellite observation.'
        : 'Satellite scenes were found. Select a product to use it in the analysis pipeline.');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Live search is unavailable.');
    } finally {
      setLiveLoading(false);
    }
  }

  async function handleSatelliteDownload() {
    if (!selectedProductId) {
      setError('Select a satellite product before downloading.');
      return;
    }
    setLiveLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await downloadSatellite({ product_id: selectedProductId, aoi });
      setDataSource('local');
      await refreshDataset();
      if (response.provider === 'mock') {
        setNotice('Synthetic fixture cached separately. It has no acquisition date or real-world coordinates and was not added to your analysis dataset.');
      } else if (response?.download?.success) {
        setNotice(`Scene ${selectedProductId} downloaded and cached successfully.`);
      } else {
        setNotice(`Product ${selectedProductId} was not downloaded.`);
      }
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Satellite product download failed.');
    } finally {
      setLiveLoading(false);
    }
  }

  const datasetAvailable = Boolean(dataset?.current?.available);
  const completedResults = useMemo(
    () => recentResults.filter((item) => !item.status || item.status === 'completed'),
    [recentResults],
  );

  const handleViewHistoryInGis = useCallback((id) => {
    setDataSource('local');
    setGisRequest((previous) => ({
      id,
      requestId: (previous?.requestId ?? 0) + 1,
    }));
    window.setTimeout(() => {
      document.querySelector('.gis-workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  }, []);

  const handleDemoMode = useCallback(async () => {
    setDemoLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await getGeoAIDemoScenario();
      setDemoSummary(response);
      setIsDemoMode(true);
      setNotice('Synthetic demo loaded. These values are not satellite observations and are not saved as an analysis.');
    } catch (requestError) {
      setDemoSummary(null);
      setIsDemoMode(false);
      setError(friendlyError(requestError));
    } finally {
      setDemoLoading(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    setIsDemoMode(false);
    setDemoSummary(null);
    setError('');
    setNotice('Workspace reset. The app is ready to start a fresh local analysis.');
    setAnalyses({});
    setResultId(null);
    setSatelliteResults([]);
    setSelectedProductId('');
    setBusy('');
    setDataSource('local');
  }, []);

  function handleAuthenticated(user) {
    setAuthUser(user);
    setAuthNotice('');
    setAuthMode('ready');
    setError('');
  }

  async function handleSignOut() {
    setSigningOut(true);
    try {
      await signOut();
      clearPrivateWorkspace();
      setAuthUser(null);
      setAuthNotice('');
      setAuthMode(authRequired ? 'login' : 'ready');
    } catch (signOutError) {
      setError(signOutError.message || 'Unable to sign out.');
    } finally {
      setSigningOut(false);
    }
  }

  if (authMode === 'checking') {
    return <main className="auth-shell"><p role="status">Checking secure workspace access…</p></main>;
  }
  if (authMode === 'login' || authMode === 'configuration-error') {
    return (
      <AuthPanel
        onAuthenticated={handleAuthenticated}
        configurationError={authMode === 'configuration-error'}
        sessionNotice={authNotice}
      />
    );
  }
  if (authMode === 'unavailable') {
    return (
      <main className="auth-shell">
        <div className="auth-card">
          <p className="eyebrow">WORKSPACE UNAVAILABLE</p>
          <h1>Unable to verify access</h1>
          <p>{backendError || 'The backend could not verify authentication status.'}</p>
          <button className="secondary-button" type="button" onClick={() => window.location.reload()}>
            Try again
          </button>
        </div>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top"><span className="brand-mark">S</span><span>ORBITAL<span className="brand-light"> / INTEL</span></span></a>
        <p className="nav-label">WORKSPACE</p>
        <a className="nav-item active" href="#overview"><span>◫</span> Dashboard</a>
        <a className="nav-item" href="#dataset"><span>▤</span> Dataset</a>
        <a className="nav-item" href="#analysis"><span>◎</span> Analysis</a>
        <a className="nav-item" href="#results"><span>↗</span> Results</a>
        {authUser && <a className="nav-item" href="#monitoring"><span>◷</span> Monitoring</a>}
        <a className="nav-item" href="#about"><span>ⓘ</span> Methodology</a>
        <div className="sidebar-bottom">
          <span className={`status-dot ${backendStatus === 'online' ? 'online' : ''}`} />
          <div><b>{backendStatus === 'online' ? 'System Online' : backendStatus === 'checking' ? 'Checking backend' : 'Backend Offline'}</b><small>Stage A · Local imagery</small></div>
        </div>
      </aside>

      <main id="top" className="main-content">
        <Header
          backendStatus={backendStatus}
          userEmail={authUser?.email}
          onSignOut={authUser ? handleSignOut : null}
          signingOut={signingOut}
        />
        {authUser && (
          <section id="monitoring">
            <SatelliteMonitoringPanel />
          </section>
        )}
        <section id="overview" className="page-heading">
          <div><p className="eyebrow">EARTH OBSERVATION / ANALYSIS</p><h1>Satellite Intelligence</h1><p>Spectral indices and a transparent land-cover baseline from GeoTIFF inputs.</p></div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button type="button" className="secondary-button" onClick={handleDemoMode} disabled={demoLoading || backendStatus !== 'online'}>
              {demoLoading ? 'Loading demo…' : isDemoMode ? 'Refresh demo' : 'Demo mode'}
            </button>
            <button type="button" className="secondary-button" onClick={handleReset}>Reset</button>
            {resultLoading && <LoadingIndicator label="Loading saved result" />}
          </div>
        </section>

        {demoLoading && <div className="alert info" role="status">Loading deterministic synthetic demo…</div>}
        {demoSummary && (
          <section className="panel" aria-label="Synthetic demo scenario">
            <div className="panel-header">
              <div><p className="eyebrow">SYNTHETIC DEMONSTRATION ONLY</p><h2>Index calculation example</h2></div>
              <span className="stage-badge">NOT SATELLITE DATA</span>
            </div>
            <p>{demoSummary.message}</p>
            <div className="ndvi-stat-grid">
              {['ndvi', 'ndwi', 'ndbi'].map((key) => (
                <div className="ndvi-stat" key={key}>
                  <small>{key.toUpperCase()} mean</small>
                  <b>{number(demoSummary.results?.[key]?.statistics?.mean)}</b>
                </div>
              ))}
              <div className="ndvi-stat">
                <small>Land-cover baseline</small>
                <b>{Object.values(demoSummary.results?.landcover?.class_distribution ?? {}).length} rule classes</b>
              </div>
            </div>
            <p className="panel-note">
              This in-memory fixture is not saved, has no acquisition date or geographic coordinates,
              and cannot be confused with a user upload or live observation.
            </p>
          </section>
        )}

        {backendStatus === 'offline' && <ErrorMessage message={backendError} />}
        {error && backendStatus !== 'offline' && <ErrorMessage message={error} onDismiss={() => setError('')} />}
        {notice && <div className="alert info" role="status"><span>i</span><p>{notice}</p><button onClick={() => setNotice('')} aria-label="Dismiss notice">×</button></div>}

        <section id="dataset">
          <div className="panel dataset-panel" aria-label="Data source selector">
            <div className="panel-header">
              <div><p className="eyebrow">DATA SOURCE</p><h2>Satellite data mode</h2></div>
            </div>
            <div className="analysis-actions" style={{ paddingTop: 12 }}>
              <button type="button" className={`secondary-button ${dataSource === 'local' ? 'selected' : ''}`} onClick={() => setDataSource('local')}>Local Dataset</button>
              <button type="button" className={`secondary-button ${dataSource === 'live' ? 'selected' : ''}`} onClick={() => setDataSource('live')}>Live Satellite</button>
            </div>
            {dataSource === 'local' ? (
              <DatasetSelector
                dataset={dataset}
                loading={datasetLoading}
                uploadingPeriod={uploadingPeriod ? `Uploading ${uploadProgress}%` : ''}
                onUpload={handleUpload}
                onRefresh={refreshDataset}
                savedResults={completedResults}
                resultsLoading={resultsLoading}
                resultsError={resultsError}
                hasMoreResults={nextResultsOffset !== null}
                loadingMoreResults={loadingOlderResults}
                onRefreshResults={refreshResults}
                onLoadMoreResults={loadOlderResults}
                onAnalysisCompleted={refreshResults}
                requestedGisResult={gisRequest}
                requestedGisScene={requestedGisScene}
              />
            ) : (
              <div className="panel" style={{ margin: '0 15px 12px', padding: 14, border: '1px solid #2b3a3c', borderRadius: 8 }}>
                <div className="panel-header">
                  <div><p className="eyebrow">LIVE SATELLITE DATA</p><h2>Discover scenes</h2></div>
                </div>
                <p className="processing-note" style={{ margin: '0 0 10px' }}>{satelliteStatus.message}</p>
                <SatelliteCatalogSearch
                  onSearchProvider={handleSatelliteSearch}
                  initialSelectedScene={selectedCatalogScene}
                  onSceneSelected={setSelectedCatalogScene}
                  onSceneAcquired={(scene) => {
                    const acquired = {
                      ...scene,
                      id: scene.id,
                      metadata: scene.metadata,
                    };
                    setRequestedGisScene(acquired);
                    setDataSource('local');
                    setNotice(`Copernicus scene ${scene.catalog_acquisition?.item_id ?? ''} was acquired and saved as ${scene.id}.`);
                    refreshDataset();
                  }}
                />
                {satelliteResults.length > 0 && (
                  <div style={{ display: 'grid', gap: 8, marginTop: 16 }}>
                    <h3 style={{ margin: 0, fontSize: 12, letterSpacing: '.12em', color: '#819294' }}>AVAILABLE SCENES</h3>
                    {satelliteResults.map((product) => (
                      <div key={product.product_id} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, padding: 10, border: '1px solid #2b3a3c', borderRadius: 6, background: '#172124' }}>
                        <div>
                          <div style={{ fontWeight: 700 }}>
                            {product.metadata?.data_classification === 'synthetic'
                              ? 'Synthetic fixture · no acquisition date'
                              : product.acquisition_date ?? 'Acquisition date unavailable'}
                          </div>
                          <small style={{ color: '#8fa0a0' }}>
                            {product.metadata?.data_classification === 'synthetic'
                              ? 'Not an observation · no cloud or geographic metadata'
                              : `${product.cloud_cover ?? 'N/A'}% cloud · ${product.platform}`}
                          </small>
                          <div style={{ color: '#b7c7c4', fontSize: 11 }}>{product.product_id}</div>
                        </div>
                        <button type="button" className="secondary-button" onClick={() => setSelectedProductId(product.product_id)}>{selectedProductId === product.product_id ? 'Selected' : 'Select'}</button>
                      </div>
                    ))}
                    {selectedProductId && (
                      <button type="button" className="primary-button" onClick={handleSatelliteDownload} disabled={liveLoading}>
                        {satelliteResults.find((product) => product.product_id === selectedProductId)?.metadata?.data_classification === 'synthetic'
                          ? 'CACHE SYNTHETIC FIXTURE'
                          : 'USE THIS SCENE'}
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section id="analysis"><AnalysisPanel busy={busy || uploadingPeriod} onRun={handleAnalysis} onRunAll={handleAllAnalysis} datasetAvailable={datasetAvailable && backendStatus === 'online' && !uploadingPeriod} /></section>

        <section id="map" className="map-panel panel">
          <div className="panel-header">
            <div><p className="eyebrow">SPATIAL VIEW</p><h2>Analysis map</h2></div>
            <div className="layer-controls" aria-label="Map layers">
              {layerConfig.map((layer) => (
                <button key={layer.key} className={`layer-button ${activeLayer === layer.key ? 'selected' : ''}`} onClick={() => setActiveLayer(layer.key)} disabled={!analyses[layer.key]}>
                  <i style={{ backgroundColor: layer.color }} />{layer.label}
                </button>
              ))}
            </div>
          </div>
          {busy && <div className="map-loading"><LoadingIndicator label="Processing satellite raster…" /></div>}
          {selected ? (
            <Suspense fallback={<div className="map-loading" role="status">Loading map module…</div>}>
              <MapView
                result={selected}
                layerName={mapLayer?.label}
                resultId={resultId}
                layerKey={activeLayer}
              />
            </Suspense>
          ) : (
            <div className="map-empty">
              <div className="map-crosshair">◎</div>
              <b>Georeferenced results will appear here</b>
              <span>No map location is assumed. Load valid rasters and run an analysis.</span>
            </div>
          )}
          <div className="map-footer"><Legend layer={activeLayer} /><span>{selected?.statistics?.valid_pixels?.toLocaleString() ?? '--'} valid pixels</span></div>
        </section>

        <section className="metric-grid" aria-label="Key metrics">
          <MetricCard title="NDVI mean" icon="⌁" value={number(current?.statistics?.mean)} subtitle={`Min ${number(current?.statistics?.min)} · Max ${number(current?.statistics?.max)}`} />
          <MetricCard title="NDWI mean" icon="≈" value={number(ndwi?.statistics?.mean)} subtitle={`Water-related indicator · ${percent(ndwi?.water_related_percentage)}`} />
          <MetricCard title="NDBI mean" icon="⌂" value={number(ndbi?.statistics?.mean)} subtitle={`Built-up indicator · ${percent(ndbi?.built_up_indicator_percentage)}`} />
          <MetricCard title="Vegetation share" icon="↗" value={percent(current?.vegetation_percentage)} subtitle={`Threshold ${number(current?.vegetation_threshold, 2)} · NDVI`} />
        </section>
        <section className="metric-grid secondary-metrics">
          <MetricCard title="Valid NDVI pixels" icon="▦" value={current?.statistics?.valid_pixels?.toLocaleString() ?? '--'} subtitle="Nodata excluded" />
          <MetricCard title="Water-related share" icon="≈" value={percent(ndwi?.water_related_percentage)} subtitle="NDWI threshold indicator" />
          <MetricCard title="Built-up indicator" icon="⌂" value={percent(ndbi?.built_up_indicator_percentage)} subtitle="Not definitive building detection" />
          <MetricCard title="Mean NDVI change" icon="Δ" value={number(change?.mean_change)} subtitle={change ? `Changed pixels ${percent(change.changed_pixel_percentage)}` : 'Historical data required'} />
        </section>

        <section className="panel" aria-label="Analysis data provenance">
          <div className="panel-header">
            <div><p className="eyebrow">INPUT PROVENANCE</p><h2>Source and data quality</h2></div>
          </div>
          {Object.keys(selectedProvenance).length ? (
            <div className="content-grid">
              {Object.entries(selectedProvenance).map(([datasetId, provenance]) => (
                <div className="ndvi-stat" key={datasetId}>
                  <strong>{datasetId} · {provenance.data_classification?.replaceAll('_', ' ')}</strong>
                  <div>Source: {provenance.source ?? 'unknown'}</div>
                  <div>Platform / sensor: {provenance.platform ?? 'unverified'} / {provenance.sensor ?? 'unverified'}</div>
                  <div>Acquisition: {provenance.acquisition_date ?? 'not recorded'}</div>
                  <div>CRS: {provenance.crs ?? 'unknown'} · Cloud cover: {provenance.cloud_cover_percentage ?? 'unknown'}</div>
                  <div>
                    Resolution:{' '}
                    {Object.values(provenance.bands ?? {})[0]?.resolution?.join(' × ') ?? 'unknown'}
                    {' '}{Object.values(provenance.bands ?? {})[0]?.resolution_units ?? ''}
                  </div>
                  {provenance.quality_warnings?.length > 0 && (
                    <ul>{provenance.quality_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="chart-empty">Run or select a saved analysis to view provenance recorded from its input rasters.</div>
          )}
          <p className="panel-note">Band names do not verify sensor identity, calibration, atmospheric correction, or cloud masking.</p>
        </section>

        <section className="panel" aria-label="Predictive GeoAI dashboard">
          <div className="panel-header">
            <div><p className="eyebrow">PREDICTIVE GEOAI</p><h2>Vegetation and risk outlook</h2></div>
            <div className="processing-note" style={{ fontSize: 11 }}>
              {geoAIStatus?.available ? `${geoAIStatus.status.toUpperCase()} · ${geoAIStatus.components?.length ?? 0} components active` : 'GeoAI unavailable'}
            </div>
          </div>
          <div className="content-grid" style={{ marginTop: 12 }}>
            <div className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">FORECAST</p><h2>NDVI outlook</h2></div></div>
              {geoAIInsights.forecast?.success ? (
                <div className="ndvi-stat-grid">
                  <div className="ndvi-stat"><small>Trend</small><b>{number(geoAIInsights.forecast?.historical_trend?.slope_per_day, 4)}</b></div>
                  <div className="ndvi-stat"><small>Latest</small><b>{number(geoAIInsights.forecast?.forecast_values?.at(-1), 3)}</b></div>
                  <div className="ndvi-stat"><small>Horizon</small><b>{geoAIInsights.forecast?.forecast_dates?.length ?? 0} days</b></div>
                  <div className="ndvi-stat"><small>Interval support</small><b>{geoAIInsights.forecast?.prediction_interval?.support ? 'Yes' : 'Limited'}</b></div>
                </div>
              ) : (
                <div className="chart-empty">
                  {geoAIInsights.forecast?.message ?? 'Forecast requires at least two distinct, dated NDVI observations.'}
                  {geoAIInsights.forecast?.observation_warnings?.length > 0
                    && <ul>{geoAIInsights.forecast.observation_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
                </div>
              )}
              {geoAIInsights.forecast?.success && (
                <p className="panel-note">
                  Input classification: {geoAIInsights.forecast.data_classification?.replaceAll('_', ' ') ?? 'unknown'}.
                  {geoAIInsights.forecast.observation_warnings?.length > 0
                    && ` ${geoAIInsights.forecast.observation_warnings.join(' ')}`}
                </p>
              )}
              <p className="panel-note">{geoAIInsights.forecast?.limitations ?? 'No forecast generated yet.'}</p>
            </div>

            <div className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">HISTORY</p><h2>Observation summary</h2></div></div>
              {geoAIInsights.history?.success ? (
                <div className="ndvi-stat-grid">
                  <div className="ndvi-stat"><small>Mean NDVI</small><b>{number(geoAIInsights.history?.mean_ndvi, 3)}</b></div>
                  <div className="ndvi-stat"><small>Latest</small><b>{number(geoAIInsights.history?.latest_ndvi, 3)}</b></div>
                  <div className="ndvi-stat"><small>Samples</small><b>{geoAIInsights.history?.count ?? 0}</b></div>
                  <div className="ndvi-stat"><small>Period</small><b>{geoAIInsights.history?.start_date ?? '--'} → {geoAIInsights.history?.end_date ?? '--'}</b></div>
                </div>
              ) : (
                <div className="chart-empty">
                  {geoAIInsights.history?.message ?? 'Historic observations are unavailable or insufficient for analysis.'}
                  {geoAIInsights.history?.observation_warnings?.length > 0
                    && <ul>{geoAIInsights.history.observation_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
                </div>
              )}
              {geoAIInsights.history?.success && (
                <p className="panel-note">
                  Input classification: {geoAIInsights.history.data_classification?.replaceAll('_', ' ') ?? 'unknown'}.
                </p>
              )}
            </div>

            <div className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">TRANSITIONS</p><h2>Land-cover change</h2></div></div>
              {geoAIInsights.transitions?.success ? (
                <>
                  <div className="ndvi-stat-grid">
                  <div className="ndvi-stat"><small>Valid pixels</small><b>{geoAIInsights.transitions?.valid_pixel_count?.toLocaleString() ?? '--'}</b></div>
                  <div className="ndvi-stat"><small>Transitions</small><b>{geoAIInsights.transitions?.transitions?.length ?? 0}</b></div>
                  <div className="ndvi-stat"><small>Largest shift</small><b>{geoAIInsights.transitions?.transitions?.[0]?.pixel_count ?? 0}</b></div>
                  <div className="ndvi-stat"><small>Source</small><b>{geoAIInsights.transitions?.transitions?.[0]?.source_label ?? 'n/a'}</b></div>
                  </div>
                  <p className="panel-note">
                    One historical-to-current raster pair; differences are not a repeated trend.
                    {' '}Classification: {geoAIInsights.transitions?.provenance?.data_classification?.replaceAll('_', ' ') ?? 'unverified inputs'}.
                  </p>
                </>
              ) : (
                <div className="chart-empty">Land-cover transitions are unavailable without aligned historical and current rasters.</div>
              )}
            </div>

            <div className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">RISK</p><h2>Indicator summary</h2></div></div>
              {geoAIInsights.risk?.success ? (
                <>
                  <div className="ndvi-stat-grid">
                  <div className="ndvi-stat"><small>Screening indicators</small><b>{geoAIInsights.risk?.count ?? 0}</b></div>
                  <div className="ndvi-stat"><small>Risk score</small><b>Not calibrated</b></div>
                  <div className="ndvi-stat"><small>Evidence</small><b>{geoAIInsights.risk?.indicators?.[0]?.indicator_name ?? 'No configured threshold triggered'}</b></div>
                  <div className="ndvi-stat"><small>Input quality</small><b>{geoAIInsights.risk?.indicators?.[0]?.data_quality?.status ?? 'See provenance'}</b></div>
                  </div>
                  <p className="panel-note">
                    Input classification: {geoAIInsights.risk.data_classification?.replaceAll('_', ' ') ?? 'unknown'}.
                    {' '}Threshold screening only; no calibrated risk score or causal explanation is produced.
                  </p>
                </>
              ) : (
                <div className="chart-empty">{geoAIInsights.risk?.message ?? 'Risk screening is unavailable without valid current raster inputs.'}</div>
              )}
            </div>

            <div className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">EVALUATION</p><h2>Forecast baseline hold-out</h2></div></div>
              {geoAIInsights.evaluation?.success ? (
                <>
                  <div className="ndvi-stat-grid">
                    <div className="ndvi-stat"><small>MAE</small><b>{number(geoAIInsights.evaluation.metrics?.mae)}</b></div>
                    <div className="ndvi-stat"><small>RMSE</small><b>{number(geoAIInsights.evaluation.metrics?.rmse)}</b></div>
                    <div className="ndvi-stat"><small>Persistence MAE</small><b>{number(geoAIInsights.evaluation.metrics?.baseline_mae)}</b></div>
                    <div className="ndvi-stat"><small>Test observations</small><b>{geoAIInsights.evaluation.test_observation_count ?? '--'}</b></div>
                  </div>
                  <p className="panel-note">{geoAIInsights.evaluation.limitations}</p>
                </>
              ) : (
                <div className="chart-empty">
                  {geoAIInsights.evaluation?.message ?? 'Evaluation is unavailable until enough distinct, metadata-dated observations exist.'}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel chart-panel ndvi-panel">
            <div className="panel-header"><div><p className="eyebrow">VEGETATION INDEX</p><h2>NDVI statistics</h2></div></div>
            {current ? <div className="ndvi-stat-grid">
              {[
                ['Mean', current.statistics?.mean],
                ['Minimum', current.statistics?.min],
                ['Maximum', current.statistics?.max],
                ['Median', current.statistics?.median],
                ['Valid pixels', current.statistics?.valid_pixels],
                ['Vegetation share', current.vegetation_percentage],
              ].map(([label, value]) => <div className="ndvi-stat" key={label}><small>{label}</small><b>{label === 'Valid pixels' ? (value?.toLocaleString() ?? '--') : label === 'Vegetation share' ? percent(value) : number(value)}</b></div>)}
              <div className="ndvi-legend"><span>Very low</span><i /><span>Low</span><i /><span>Moderate</span><i /><span>High</span><i /><span>Very high</span></div>
            </div> : <div className="chart-empty">NDVI statistics will appear after calculating B04 and B08.</div>}
            <p className="panel-note">Index ranges are contextual indicators, not universal crop-health categories.</p>
          </div>
          {landcover ? (
            <Suspense fallback={<div className="panel chart-panel chart-empty" role="status">Loading land-cover chart…</div>}>
              <LandCoverChart result={landcover} />
            </Suspense>
          ) : (
            <article id="landcover" className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">BASELINE CLASSIFICATION</p><h2>Land-cover distribution</h2></div><span className="method-badge">HEURISTIC</span></div>
              <div className="chart-empty">Land-cover values appear here after a successful classification.</div>
              <p className="panel-note">Transparent index-threshold baseline; not a trained model or ground truth.</p>
            </article>
          )}
          {change ? (
            <Suspense fallback={<div className="panel chart-panel chart-empty" role="status">Loading vegetation-change chart…</div>}>
              <ChangeChart result={change} currentResult={current} />
            </Suspense>
          ) : (
            <article id="change" className="panel chart-panel">
              <div className="panel-header"><div><p className="eyebrow">TEMPORAL COMPARISON</p><h2>Vegetation change</h2></div><span className="method-badge">HISTORICAL DATA NEEDED</span></div>
              <div className="chart-empty">Aligned historical B04 and B08 imagery is required to compare periods.</div>
              <p className="panel-note">Change = current NDVI − historical NDVI. Not a causal assessment.</p>
            </article>
          )}
          <InsightPanel messages={insights} />
        </section>

        <section id="results" className="bottom-grid">
          <section className="panel downloads-panel">
            <div className="panel-header">
              <div><p className="eyebrow">SAVED OUTPUTS</p><h2>Recent results</h2></div>
              <div className="result-controls">
                <select className="result-select" aria-label="Load saved result" onChange={handleSelectResult} value="">
                  <option value="">Select a saved result</option>
                  {completedResults.map((item) => <option value={item.id} key={item.id}>{item.analysis} · {item.created_at ? new Date(item.created_at).toLocaleString() : 'date unknown'}</option>)}
                </select>
                {nextResultsOffset !== null && (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={loadOlderResults}
                    disabled={loadingOlderResults}
                  >
                    {loadingOlderResults ? 'Loading…' : 'Load older results'}
                  </button>
                )}
              </div>
            </div>
            {resultId && <p className="panel-note">Current result ID: <code>{resultId}</code></p>}
            <DownloadPanel
              resultId={resultId}
              artifacts={resultArtifacts}
              loading={artifactsLoading}
              error={artifactsError}
              onError={setError}
            />
          </section>
          <article className="panel about-panel" id="about">
            <div className="panel-header"><div><p className="eyebrow">METHOD</p><h2>How to interpret results</h2></div></div>
            <p>Local GeoTIFF bands are validated and masked for nodata before spectral indices are calculated. A transparent index-threshold baseline supports land-cover categories; aligned historical rasters enable NDVI comparison.</p>
            <div className="formula-list">
              <span><b>NDVI</b> (B08 − B04) / (B08 + B04)</span>
              <span><b>NDWI</b> McFeeters: (B03 − B08) / (B03 + B08)</span>
              <span><b>NDBI</b> (B11 − B08) / (B11 + B08)</span>
            </div>
            <p className="panel-note">Satellite-derived indicators are not ground truth. Validate with field observations where needed. Stage B/live imagery is not enabled.</p>
          </article>
        </section>

        <AnalysisHistory
          key={authUser?.id ?? 'public'}
          results={recentResults}
          loading={resultsLoading}
          error={resultsError}
          hasMore={nextResultsOffset !== null}
          loadingMore={loadingOlderResults}
          onRefresh={refreshResults}
          onLoadMore={loadOlderResults}
          onViewInGis={handleViewHistoryInGis}
        />

        <footer className="footer"><span>ORBITAL / INTEL</span><span>Scientific indicators · Stage A offline processing</span><span>Backend: {backendStatus === 'online' ? 'Connected' : backendStatus === 'checking' ? 'Checking' : 'Offline'}</span></footer>
      </main>
    </div>
  );
}

export default App;
