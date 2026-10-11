import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SatelliteCatalogSearch from './SatelliteCatalogSearch';

const service = vi.hoisted(() => ({
  acquireCopernicusScene: vi.fn(),
  searchCopernicusCatalog: vi.fn(),
}));

vi.mock('../services/api', () => service);

const scene = {
  id: 'S2C_TEST_PRODUCT',
  collection: 'sentinel-2-l2a',
  satellite: 'sentinel-2a',
  acquisition_datetime: '2025-01-27T05:10:09.024Z',
  cloud_cover: 6.94,
  bbox: [77, 12, 77.1, 12.1],
  footprint: null,
  processing_level: 'S2MSI2A',
  catalog_url: 'https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/S2A_TEST_PRODUCT',
  provider: 'Copernicus Data Space Ecosystem',
  available_assets: [
    { key: 'B02_10m', title: 'Blue (band 2) - 10m', gsd: 10, size_bytes: 100 },
    { key: 'B03_10m', title: 'Green (band 3) - 10m', gsd: 10, size_bytes: 200 },
    { key: 'B04_10m', title: 'Red (band 4) - 10m', gsd: 10, size_bytes: 300 },
    { key: 'B08_10m', title: 'NIR (band 8) - 10m', gsd: 10, size_bytes: 400 },
  ],
  acquisition_supported: true,
  acquisition_bytes: 1000,
};

function fillSearchForm() {
  fireEvent.change(screen.getByLabelText('AOI GeoJSON'), {
    target: {
      value: JSON.stringify({
        type: 'Polygon',
        coordinates: [[[77, 12], [77, 12.1], [77.1, 12.1], [77.1, 12], [77, 12]]],
      }),
    },
  });
  fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2025-01-01' } });
  fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2025-01-31' } });
}

beforeEach(() => {
  vi.clearAllMocks();
  service.acquireCopernicusScene.mockResolvedValue({
    id: 'saved-scene-1',
    metadata: { catalog_source: { item_id: scene.id } },
    catalog_acquisition: { item_id: scene.id },
  });
});

describe('SatelliteCatalogSearch', () => {
  it('shows loading state, results and selected scene handoff metadata', async () => {
    let finishSearch;
    const onSceneSelected = vi.fn();
    service.searchCopernicusCatalog.mockImplementation(
      () => new Promise((resolve) => { finishSearch = resolve; }),
    );
    render(<SatelliteCatalogSearch onSceneSelected={onSceneSelected} />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));

    expect(screen.getByText('Searching the official Copernicus catalog…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Searching Copernicus…' })).toBeDisabled();
    finishSearch({ results: [scene], next_cursor: null });

    expect(await screen.findByText(scene.id)).toBeInTheDocument();
    expect(screen.getByText(/6.94% cloud/)).toBeInTheDocument();
    expect(screen.getByText('STAC-listed spectral assets (4)')).toBeInTheDocument();
    expect(screen.getByText(/estimated full-asset transfer/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Select scene' }));

    expect(onSceneSelected).toHaveBeenCalledWith(scene);
    expect(screen.getByRole('status', { name: 'Selected catalog scene' })).toHaveTextContent(scene.id);
    expect(screen.getByRole('link', { name: 'Open Copernicus catalog item' })).toHaveAttribute(
      'href',
      scene.catalog_url,
    );
    expect(screen.getByText(/No product assets have been fetched/)).toBeInTheDocument();
    expect(service.searchCopernicusCatalog).toHaveBeenCalledWith({
      aoi: JSON.parse(screen.getByLabelText('AOI GeoJSON').value),
      start_date: '2025-01-01',
      end_date: '2025-01-31',
      max_cloud_cover: 20,
      limit: 20,
    });
  });

  it('shows an empty-result state', async () => {
    service.searchCopernicusCatalog.mockResolvedValue({ results: [], next_cursor: null });
    render(<SatelliteCatalogSearch />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));

    expect(await screen.findByText('No Sentinel-2 scenes matched this search.')).toBeInTheDocument();
  });

  it('shows a safe error and retries the same search', async () => {
    service.searchCopernicusCatalog
      .mockRejectedValueOnce({
        response: { status: 503, data: { detail: 'The Copernicus catalog is temporarily unavailable.' } },
      })
      .mockResolvedValueOnce({ results: [scene], next_cursor: null });
    render(<SatelliteCatalogSearch />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('temporarily unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry search' }));
    expect(await screen.findByText(scene.id)).toBeInTheDocument();
    expect(service.searchCopernicusCatalog).toHaveBeenCalledTimes(2);
  });

  it('appends another page using the opaque cursor', async () => {
    service.searchCopernicusCatalog
      .mockResolvedValueOnce({ results: [scene], next_cursor: 'opaque-cursor' })
      .mockResolvedValueOnce({
        results: [{ ...scene, id: 'S2B_NEXT_PRODUCT' }],
        next_cursor: null,
      });
    render(<SatelliteCatalogSearch />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));
    await screen.findByText(scene.id);
    fireEvent.click(screen.getByRole('button', { name: 'Load more scenes' }));

    expect(await screen.findByText('S2B_NEXT_PRODUCT')).toBeInTheDocument();
    expect(service.searchCopernicusCatalog).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ cursor: 'opaque-cursor' }),
    );
  });

  it('keeps configured-provider search available without acquiring catalog scenes', async () => {
    service.searchCopernicusCatalog.mockResolvedValue({ results: [] });
    const onSearchProvider = vi.fn();
    render(<SatelliteCatalogSearch onSearchProvider={onSearchProvider} />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search configured provider' }));

    await waitFor(() => expect(onSearchProvider).toHaveBeenCalledWith({
      aoi: JSON.parse(screen.getByLabelText('AOI GeoJSON').value),
      start_date: '2025-01-01',
      end_date: '2025-01-31',
      max_cloud_cover: 20,
    }));
    expect(service.searchCopernicusCatalog).not.toHaveBeenCalled();
  });

  it('acquires a selected scene and hands the saved GeoTIFF scene to the GIS workspace', async () => {
    service.searchCopernicusCatalog.mockResolvedValue({ results: [scene] });
    const onSceneAcquired = vi.fn();
    render(<SatelliteCatalogSearch onSceneAcquired={onSceneAcquired} />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));
    await screen.findByText(scene.id);
    fireEvent.click(screen.getByRole('button', { name: 'Select scene' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download and ingest selected scene' }));

    expect(await screen.findByText(/downloaded and ingested as a GeoTIFF scene/)).toBeInTheDocument();
    expect(service.acquireCopernicusScene).toHaveBeenCalledWith({
      provider: scene.provider,
      collection: scene.collection,
      scene_id: scene.id,
      aoi: JSON.parse(screen.getByLabelText('AOI GeoJSON').value),
    });
    expect(onSceneAcquired).toHaveBeenCalledWith(expect.objectContaining({
      id: 'saved-scene-1',
      catalog_acquisition: { item_id: scene.id },
    }));
  });

  it('shows a distinct in-progress state while imagery is being acquired', async () => {
    service.searchCopernicusCatalog.mockResolvedValue({ results: [scene] });
    let finishAcquisition;
    service.acquireCopernicusScene.mockImplementation(
      () => new Promise((resolve) => { finishAcquisition = resolve; }),
    );
    render(<SatelliteCatalogSearch />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));
    await screen.findByText(scene.id);
    fireEvent.click(screen.getByRole('button', { name: 'Select scene' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download and ingest selected scene' }));

    expect(screen.getByText(/Downloading B02, B03, B04 and B08 at 10 m/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Downloading and ingesting…' })).toBeDisabled();
    finishAcquisition({ id: 'saved-scene-1', metadata: {} });
    expect(await screen.findByText(/downloaded and ingested as a GeoTIFF scene/)).toBeInTheDocument();
  });

  it('shows an acquisition failure and allows a retry', async () => {
    service.searchCopernicusCatalog.mockResolvedValue({ results: [scene] });
    service.acquireCopernicusScene
      .mockRejectedValueOnce({
        response: { status: 503, data: { detail: 'Configure CDSE credentials on the backend.' } },
      })
      .mockResolvedValueOnce({ id: 'saved-scene-1', metadata: {} });
    render(<SatelliteCatalogSearch />);
    fillSearchForm();
    fireEvent.click(screen.getByRole('button', { name: 'Search Copernicus scenes' }));
    await screen.findByText(scene.id);
    fireEvent.click(screen.getByRole('button', { name: 'Select scene' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download and ingest selected scene' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Configure CDSE credentials');
    fireEvent.click(screen.getByRole('button', { name: 'Retry acquisition' }));
    expect(service.acquireCopernicusScene).toHaveBeenCalledTimes(2);
  });
});
