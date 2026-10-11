import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ImageryIngestionPanel from './ImageryIngestionPanel';
import {
  analyzeImageryScene,
  downloadArtifact,
  getImageryAnalysis,
  getImageryAnalysisTypes,
  getImageryScene,
  getAIModels,
  getResult,
  getResultArtifacts,
  getResults,
  ingestImagery,
  listImageryScenes,
} from '../services/api';

vi.mock('../services/api', () => ({
  analyzeImageryScene: vi.fn(),
  downloadArtifact: vi.fn(),
  downloadResult: vi.fn(),
  getImageryAnalysis: vi.fn(),
  getImageryAnalysisTypes: vi.fn(),
  getImageryScene: vi.fn(),
  getAIModels: vi.fn(),
  getAIInferenceResult: vi.fn(),
  runAIInference: vi.fn(),
  getResult: vi.fn(),
  getResultArtifacts: vi.fn(),
  getResultImage: vi.fn(),
  getResults: vi.fn(),
  ingestImagery: vi.fn(),
  listImageryScenes: vi.fn(),
}));

const scene = {
  id: 'scene-1',
  persistence: { mode: 'cloud-private' },
  metadata: {
    original_filename: 'sample-scene.tif',
    acquisition_date: null,
    crs: 'EPSG:4326',
    width: 8,
    height: 6,
    band_count: 3,
    bands: [{ code: 'B04' }, { code: 'B03' }, { code: 'B02' }],
    preview: {
      description: 'RGB preview from known bands.',
      crs: 'EPSG:4326',
      bounds_wgs84: [18, 12.94, 18.08, 13],
    },
  },
};

describe('ImageryIngestionPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listImageryScenes.mockResolvedValue({ scenes: [] });
    ingestImagery.mockResolvedValue(scene);
    getImageryScene.mockResolvedValue(scene);
    getImageryAnalysisTypes.mockResolvedValue({ analysis_types: [] });
    getAIModels.mockResolvedValue({
      models: [{
        id: 'sentinel2-landcover-sklearn',
        name: 'Local classifier',
        status: 'model_not_ready',
        readiness_detail: 'No model is configured.',
        input_requirements: { bands: ['B02', 'B03', 'B04', 'B08'] },
      }],
    });
    downloadArtifact.mockResolvedValue({ data: new Blob(['preview']) });
    getResults.mockResolvedValue({ results: [], has_more: false });
    getResult.mockResolvedValue({});
    getResultArtifacts.mockResolvedValue({ artifacts: [] });
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:scene-preview'),
      revokeObjectURL: vi.fn(),
    });
  });

  it('uploads a scene, retrieves its saved preview, and displays source metadata', async () => {
    render(<ImageryIngestionPanel />);
    const file = new File(['tiff-data'], 'sample-scene.tif', { type: 'image/tiff' });
    fireEvent.change(screen.getByLabelText('Upload satellite scene GeoTIFF'), {
      target: { files: [file] },
    });

    expect(await screen.findByAltText('Preview of sample-scene.tif')).toBeInTheDocument();
    expect(screen.getByText('Not reported in source metadata')).toBeInTheDocument();
    expect(screen.getByText('EPSG:4326')).toBeInTheDocument();
    expect(screen.getByText('Private cloud storage')).toBeInTheDocument();
    expect(ingestImagery).toHaveBeenCalledWith(file, expect.any(Function));
    expect(getImageryScene).toHaveBeenCalledWith('scene-1');
    expect(downloadArtifact).toHaveBeenCalledWith('scene-1', 'preview.png');
  });

  it('shows backend validation failures without claiming the upload succeeded', async () => {
    ingestImagery.mockRejectedValue({
      response: { status: 422, data: { detail: 'The GeoTIFF has no CRS.' } },
    });
    render(<ImageryIngestionPanel />);
    fireEvent.change(screen.getByLabelText('Upload satellite scene GeoTIFF'), {
      target: { files: [new File(['invalid'], 'invalid.tif', { type: 'image/tiff' })] },
    });

    expect(await screen.findByRole('alert')).toHaveTextContent('The GeoTIFF has no CRS.');
    expect(screen.queryByText('Private cloud storage')).not.toBeInTheDocument();
  });
});
