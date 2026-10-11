import ImageryIngestionPanel from './ImageryIngestionPanel';

const BAND_CODES = ['B02', 'B03', 'B04', 'B08', 'B11'];

function Period({ period, title, dataset, uploading, onUpload }) {
  const bands = new Set(dataset?.bands?.map((band) => band.code) ?? []);
  return (
    <div className="dataset-period">
      <div className="dataset-period-heading">
        <b>{title}</b>
        <span className={`dataset-state ${dataset?.available ? 'ready' : 'missing'}`}>
          {dataset?.available ? 'AVAILABLE' : 'NOT FOUND'}
        </span>
      </div>
      <div className="dataset-bands">
        {BAND_CODES.map((code) => (
          <span className={`band-chip ${bands.has(code) ? 'found' : ''}`} key={code}>{code}</span>
        ))}
      </div>
      {dataset?.available && dataset.bands[0] && (
        <small>{dataset.bands[0].width} × {dataset.bands[0].height} · {dataset.bands[0].crs || 'CRS unavailable'}</small>
      )}
      <label className={`upload-button ${uploading ? 'uploading' : ''}`}>
        <input
          aria-label={`Upload ${title} GeoTIFF bands`}
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          disabled={Boolean(uploading)}
          onChange={(event) => {
            const selectedFiles = Array.from(event.target.files ?? []);
            event.target.value = '';
            if (selectedFiles.length) onUpload(period, selectedFiles);
          }}
        />
        {uploading || `Upload ${title} GeoTIFFs`}
      </label>
    </div>
  );
}

export default function DatasetSelector({
  dataset,
  loading,
  uploadingPeriod,
  onUpload,
  onRefresh,
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
  return (
    <section className="dataset-panel panel" aria-label="Dataset availability">
      <div className="panel-header">
        <div><p className="eyebrow">LOCAL DATA SOURCE</p><h2>Dataset availability</h2></div>
        <button className="text-button" type="button" onClick={onRefresh} disabled={loading}>
          {loading ? 'Checking…' : 'Refresh ↻'}
        </button>
      </div>
      <div className="dataset-period-grid">
        <Period period="current" title="Current dataset" dataset={dataset?.current} uploading={uploadingPeriod === 'current' ? 'Uploading…' : null} onUpload={onUpload} />
        <Period period="historical" title="Historical dataset" dataset={dataset?.historical} uploading={uploadingPeriod === 'historical' ? 'Uploading…' : null} onUpload={onUpload} />
      </div>
      <p className="upload-hint">Select GeoTIFF files named with a Sentinel-2 band token (B02, B03, B04, B08, or B11). Duplicate bands are rejected; files must match the dataset georeferencing.</p>
      {!dataset?.current?.available && (
        <div className="dataset-instructions">
          <b>No satellite dataset available.</b>
          <span>Add Sentinel-2 GeoTIFF bands to <code>backend/data/current/</code>. NDVI requires B04 and B08.</span>
        </div>
      )}
      {dataset?.current?.errors?.length > 0 && (
        <p className="dataset-warning">Some raster files could not be read. Check TIFF integrity and band filenames.</p>
      )}
      <ImageryIngestionPanel
        savedResults={savedResults}
        resultsLoading={resultsLoading}
        resultsError={resultsError}
        hasMoreResults={hasMoreResults}
        loadingMoreResults={loadingMoreResults}
        onRefreshResults={onRefreshResults}
        onLoadMoreResults={onLoadMoreResults}
        onAnalysisCompleted={onAnalysisCompleted}
        requestedGisResult={requestedGisResult}
        requestedGisScene={requestedGisScene}
      />
    </section>
  );
}
