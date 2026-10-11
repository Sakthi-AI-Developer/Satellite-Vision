import { useEffect, useState } from 'react';
import { acquireCopernicusScene, searchCopernicusCatalog } from '../services/api';

function requestError(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (error.code === 'ECONNABORTED') return 'Copernicus catalog search timed out. Retry the search.';
  if (error.response?.status === 429) return 'Copernicus is rate limiting requests. Wait briefly and retry.';
  if (!error.response) return 'Copernicus catalog is unavailable. Check the connection and retry.';
  return `Copernicus catalog search failed with HTTP ${error.response.status}.`;
}

function boundsLabel(bounds) {
  return Array.isArray(bounds) && bounds.length >= 4
    ? bounds.map((value) => Number(value).toFixed(4)).join(', ')
    : 'Not reported';
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'Size not reported';
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

export default function SatelliteCatalogSearch({
  onSearchProvider,
  initialSelectedScene = null,
  onSceneSelected,
  onSceneAcquired,
}) {
  const [aoiText, setAoiText] = useState('');
  const [dates, setDates] = useState({ start: '', end: '' });
  const [maxCloudCover, setMaxCloudCover] = useState(20);
  const [results, setResults] = useState([]);
  const [nextCursor, setNextCursor] = useState(null);
  const [selectedScene, setSelectedScene] = useState(initialSelectedScene);
  const [selectedAoi, setSelectedAoi] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [lastAttempt, setLastAttempt] = useState(null);
  const [acquiring, setAcquiring] = useState(false);
  const [acquisitionError, setAcquisitionError] = useState('');
  const [acquiredScene, setAcquiredScene] = useState(null);

  useEffect(() => {
    setSelectedScene(initialSelectedScene);
  }, [initialSelectedScene]);

  function clearSearch() {
    setResults([]);
    setNextCursor(null);
    setSelectedScene(null);
    setSelectedAoi(null);
    setAcquiredScene(null);
    setAcquisitionError('');
    onSceneSelected?.(null);
    setError('');
    setNotice('');
  }

  async function executeSearch(criteria, cursor = null, append = false) {
    const attempt = { criteria, cursor, append };
    setLastAttempt(attempt);
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await searchCopernicusCatalog({
        ...criteria,
        ...(cursor ? { cursor } : {}),
      });
      const page = response.results ?? [];
      setResults((previous) => (append ? [...previous, ...page] : page));
      setNextCursor(response.next_cursor ?? null);
      setNotice(page.length ? '' : 'No Sentinel-2 scenes matched this search.');
    } catch (requestErrorValue) {
      setError(requestError(requestErrorValue));
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    let aoi;
    try {
      aoi = JSON.parse(aoiText);
    } catch {
      setError('Enter a valid AOI GeoJSON polygon before searching.');
      return;
    }
    if (!dates.start || !dates.end) {
      setError('Select both a start date and an end date before searching.');
      return;
    }
    if (dates.start > dates.end) {
      setError('The start date must be on or before the end date.');
      return;
    }
    const criteria = {
      aoi,
      start_date: dates.start,
      end_date: dates.end,
      max_cloud_cover: Number(maxCloudCover),
      limit: 20,
    };
    setSelectedAoi(aoi);
    setSelectedScene(null);
    setAcquiredScene(null);
    setAcquisitionError('');
    onSceneSelected?.(null);
    setResults([]);
    setNextCursor(null);
    await executeSearch(criteria);
  }

  async function acquireSelectedScene() {
    if (!selectedScene || acquiring) return;
    setAcquiring(true);
    setAcquisitionError('');
    setAcquiredScene(null);
    try {
      const response = await acquireCopernicusScene({
        provider: selectedScene.provider,
        collection: selectedScene.collection,
        scene_id: selectedScene.id,
        aoi: selectedAoi,
      });
      setAcquiredScene(response);
      onSceneAcquired?.(response);
    } catch (requestErrorValue) {
      const detail = requestErrorValue.response?.data?.detail;
      setAcquisitionError(
        typeof detail === 'string'
          ? detail
          : requestErrorValue.code === 'ECONNABORTED'
            ? 'Scene acquisition timed out. Check saved scenes before retrying.'
            : !requestErrorValue.response
              ? 'The acquisition service is unavailable. Check the connection and retry.'
              : `Scene acquisition failed with HTTP ${requestErrorValue.response.status}.`,
      );
    } finally {
      setAcquiring(false);
    }
  }

  return (
    <section className="catalog-search" aria-label="Satellite Data Discovery">
      <div className="panel-header">
        <div>
          <p className="eyebrow">COPERNICUS DATA SPACE · FREE CATALOG</p>
          <h3>Sentinel-2 scene discovery</h3>
        </div>
      </div>
      <p className="processing-note">
        Search public Sentinel-2 Level-2A metadata. Selecting a result retains its
        catalog reference. Acquisition downloads the selected product's four complete
        10 m source assets, then crops the saved GeoTIFF to the searched AOI.
      </p>
      <form onSubmit={handleSubmit} className="catalog-search-form">
        <label>
          <span>AOI GeoJSON</span>
          <textarea
            aria-label="AOI GeoJSON"
            value={aoiText}
            onChange={(event) => {
              setAoiText(event.target.value);
              clearSearch();
            }}
            rows={5}
            placeholder='{"type":"Polygon","coordinates":[...]}'
          />
        </label>
        <div className="catalog-date-fields">
          <label>
            <span>Start date</span>
            <input
              type="date"
              required
              aria-label="Start date"
              value={dates.start}
              onChange={(event) => {
                setDates((previous) => ({ ...previous, start: event.target.value }));
                clearSearch();
              }}
            />
          </label>
          <label>
            <span>End date</span>
            <input
              type="date"
              required
              aria-label="End date"
              value={dates.end}
              onChange={(event) => {
                setDates((previous) => ({ ...previous, end: event.target.value }));
                clearSearch();
              }}
            />
          </label>
        </div>
        <label className="catalog-cloud-control">
          <span>Maximum cloud cover ({maxCloudCover}%)</span>
          <input
            type="range"
            min="0"
            max="100"
            value={maxCloudCover}
            onChange={(event) => {
              setMaxCloudCover(Number(event.target.value));
              clearSearch();
            }}
          />
        </label>
        <button className="primary-button" type="submit" disabled={loading}>
          {loading ? 'Searching Copernicus…' : 'Search Copernicus scenes'}
        </button>
        {onSearchProvider && (
          <button
            className="secondary-button"
            type="button"
            disabled={loading}
            onClick={() => {
              let aoi;
              try {
                aoi = JSON.parse(aoiText);
              } catch {
                setError('Enter a valid AOI GeoJSON polygon before searching.');
                return;
              }
              if (!dates.start || !dates.end || dates.start > dates.end) {
                setError('Select a valid start and end date before searching.');
                return;
              }
              onSearchProvider({
                aoi,
                start_date: dates.start,
                end_date: dates.end,
                max_cloud_cover: Number(maxCloudCover),
              });
            }}
          >
            Search configured provider
          </button>
        )}
      </form>
      {loading && <p className="processing-note" role="status">Searching the official Copernicus catalog…</p>}
      {error && (
        <div className="catalog-error" role="alert">
          <span>{error}</span>
          {lastAttempt && (
            <button
              className="secondary-button"
              type="button"
              disabled={loading}
              onClick={() => executeSearch(
                lastAttempt.criteria,
                lastAttempt.cursor,
                lastAttempt.append,
              )}
            >
              Retry search
            </button>
          )}
        </div>
      )}
      {notice && <p className="processing-note" role="status">{notice}</p>}
      {results.length > 0 && (
        <div className="catalog-results" aria-label="Catalog scene results">
          <h4>RESULTS · {results.length} shown · up to 20 per page</h4>
          {results.map((scene) => (
            <article className="catalog-scene" key={`${scene.collection}:${scene.id}`}>
              <div>
                <strong>{scene.acquisition_datetime ?? 'Acquisition date unavailable'}</strong>
                <p>
                  {scene.cloud_cover === null ? 'Cloud cover not reported' : `${scene.cloud_cover}% cloud`}
                  {' · '}
                  {scene.satellite ?? 'Satellite not reported'}
                  {scene.processing_level ? ` · ${scene.processing_level}` : ''}
                </p>
                <code>{scene.id}</code>
                <small>Footprint bounds (WGS84): {boundsLabel(scene.bbox)}</small>
                <details className="catalog-assets">
                  <summary>
                    STAC-listed spectral assets ({scene.available_assets?.length ?? 0})
                  </summary>
                  {scene.available_assets?.length ? (
                    <ul>
                      {scene.available_assets.map((asset) => (
                        <li key={asset.key}>
                          {asset.key}
                          {asset.gsd ? ` · ${asset.gsd} m` : ''}
                          {asset.title ? ` · ${asset.title}` : ''}
                          {asset.size_bytes ? ` · ${formatBytes(asset.size_bytes)}` : ''}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No spectral assets were reported by the catalog.</p>
                  )}
                </details>
                <small>
                  {scene.acquisition_supported
                    ? `Four required 10 m assets are listed · estimated full-asset transfer ${formatBytes(scene.acquisition_bytes)}.`
                    : 'The required four 10 m assets are not all listed for acquisition.'}
                </small>
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  setSelectedScene(scene);
                  setAcquiredScene(null);
                  setAcquisitionError('');
                  onSceneSelected?.(scene);
                }}
              >
                {selectedScene?.id === scene.id ? 'Selected' : 'Select scene'}
              </button>
            </article>
          ))}
          {nextCursor && (
            <button
              className="secondary-button"
              type="button"
              disabled={loading}
              onClick={() => {
                const criteria = lastAttempt?.criteria;
                if (criteria) executeSearch(criteria, nextCursor, true);
              }}
            >
              {loading ? 'Loading…' : 'Load more scenes'}
            </button>
          )}
        </div>
      )}
      {selectedScene && (
        <div className="catalog-selection" role="status" aria-label="Selected catalog scene">
          <strong>Selected catalog item</strong>
          <dl>
            <div><dt>Provider</dt><dd>{selectedScene.provider}</dd></div>
            <div><dt>Collection</dt><dd>{selectedScene.collection}</dd></div>
            <div><dt>Stable scene identifier</dt><dd>{selectedScene.id}</dd></div>
            <div>
              <dt>Catalog product reference</dt>
              <dd>
                <a href={selectedScene.catalog_url} target="_blank" rel="noreferrer">
                  Open Copernicus catalog item
                </a>
              </dd>
            </div>
          </dl>
          <p>
            {acquiredScene
              ? 'Selected product assets were downloaded and ingested as a GeoTIFF scene.'
              : 'Metadata selected only. No product assets have been fetched.'}
          </p>
          {selectedScene.acquisition_supported && (
            <p>
              The four full JP2 assets are downloaded before the output is cropped
              to this AOI; listed source sizes are estimates and the backend enforces
              its configured byte and pixel limits.
            </p>
          )}
          <p>
            Acquisition requires a signed-in app user and server-side CDSE account
            credentials. The backend revalidates the catalog item and limits total
            downloaded assets to the configured imagery size cap.
          </p>
          <button
            className="primary-button"
            type="button"
            disabled={acquiring || !selectedScene.acquisition_supported || !selectedAoi}
            onClick={acquireSelectedScene}
          >
            {acquiring ? 'Downloading and ingesting…' : 'Download and ingest selected scene'}
          </button>
          {!selectedAoi && (
            <p className="processing-note">
              Search and select a scene with an AOI before requesting acquisition.
            </p>
          )}
          {selectedAoi && !selectedScene.acquisition_supported && (
            <p className="processing-note">
              This result does not list all required 10 m assets. Choose a scene that
              lists B02, B03, B04, and B08 at 10 m.
            </p>
          )}
          {acquiring && (
            <p className="processing-note" role="status">
              Downloading B02, B03, B04 and B08 at 10 m, preparing a masked GeoTIFF,
              and saving it to your imagery library. This may take several minutes.
            </p>
          )}
          {acquisitionError && (
            <div className="catalog-error" role="alert">
              <span>{acquisitionError}</span>
              <button
                className="secondary-button"
                type="button"
                disabled={acquiring}
                onClick={acquireSelectedScene}
              >
                Retry acquisition
              </button>
            </div>
          )}
          {acquiredScene && (
            <p className="processing-note" role="status">
              Scene acquired and saved as {acquiredScene.id}. Its GeoTIFF is ready in
              the existing scene-analysis and GIS workspace.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
