import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import AIInferencePanel from './AIInferencePanel';
import {
  downloadArtifact,
  getAIInferenceResult,
  getAIModels,
  runAIInference,
} from '../services/api';

vi.mock('../services/api', () => ({
  downloadArtifact: vi.fn(),
  getAIInferenceResult: vi.fn(),
  getAIModels: vi.fn(),
  runAIInference: vi.fn(),
}));

const model = {
  id: 'sentinel2-landcover-sklearn',
  name: 'Local four-band classifier',
  version: '1.0',
  framework: 'scikit-learn',
  framework_version: '1.9',
  license: 'Example model license',
  attribution: 'Operator-supplied model',
  status: 'ready',
  input_requirements: {
    bands: ['B02', 'B03', 'B04', 'B08'],
    normalization: 'divide_by_10000',
  },
};

const completed = {
  id: 'analysis-1',
  status: 'completed',
  result: {
    model: { name: model.name, version: model.version },
    processing_time_seconds: 1.25,
    valid_pixels: 12,
    confidence: { mean: 0.9, minimum: 0.75, maximum: 1 },
    rasters: {
      classification: { width: 4, height: 3, crs: 'EPSG:32643' },
    },
    classes: [
      { class_index: 1, label: 'vegetation', pixel_count: 8, percentage: 66.7 },
      { class_index: 2, label: 'water', pixel_count: 4, percentage: 33.3 },
    ],
  },
};

describe('AIInferencePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAIModels.mockResolvedValue({ models: [model] });
    runAIInference.mockResolvedValue(completed);
    downloadArtifact.mockResolvedValue({ data: new Blob(['raster']) });
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:result'),
      revokeObjectURL: vi.fn(),
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  it('reports readiness and submits local inference with actual results and downloads', async () => {
    const onAnalysisCompleted = vi.fn();
    render(<AIInferencePanel sceneId="scene-1" onAnalysisCompleted={onAnalysisCompleted} />);

    expect(await screen.findByText(/Local four-band classifier/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Run AI land-cover inference' }));

    expect(await screen.findByRole('region', { name: 'AI inference result' })).toBeInTheDocument();
    expect(runAIInference).toHaveBeenCalledWith('scene-1', {
      model_id: 'sentinel2-landcover-sklearn',
    });
    expect(screen.getByText(/confidence is not an accuracy guarantee/i)).toBeInTheDocument();
    expect(screen.getByText(/vegetation · 8 pixels/)).toBeInTheDocument();
    await waitFor(() => expect(onAnalysisCompleted).toHaveBeenCalledWith(completed));

    fireEvent.click(screen.getByRole('button', { name: 'Download classification.tif' }));
    await waitFor(() => expect(downloadArtifact).toHaveBeenCalledWith('analysis-1', 'classification.tif'));
  });

  it('keeps inference disabled and explains when the optional model is not ready', async () => {
    getAIModels.mockResolvedValue({
      models: [{
        ...model,
        status: 'model_not_ready',
        readiness_detail: 'Configure a local model manifest and trusted weights.',
      }],
    });
    render(<AIInferencePanel sceneId="scene-2" />);

    expect(await screen.findByText('Model not ready')).toBeInTheDocument();
    expect(screen.getByText('Configure a local model manifest and trusted weights.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run AI land-cover inference' })).toBeDisabled();
    expect(runAIInference).not.toHaveBeenCalled();
  });

  it('shows Prithvi compatibility blockers without enabling or fabricating inference', async () => {
    getAIModels.mockResolvedValue({
      models: [],
      candidates: [{
        id: 'prithvi-eo-2.0-tiny-tl',
        name: 'Prithvi-EO-2.0 tiny-TL (pretrained backbone)',
        status: 'model_not_ready',
        license: 'Apache-2.0',
        source: 'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL',
        readiness_detail: 'The pretrained checkpoint requires a segmentation head.',
        input_requirements: {
          bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
          format: 'four-frame HLS sequence',
        },
      }],
    });
    render(<AIInferencePanel sceneId="scene-prithvi" />);

    expect(await screen.findByText(/Prithvi-EO-2.0 tiny-TL.*model not ready/)).toBeInTheDocument();
    expect(screen.getByText(/requires a segmentation head/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Review checkpoint and license' }))
      .toHaveAttribute('href', 'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL');
    expect(screen.getByRole('button', { name: 'Run AI land-cover inference' })).toBeDisabled();
    expect(runAIInference).not.toHaveBeenCalled();
  });

  it('shows API and inference failures without displaying a fabricated result', async () => {
    runAIInference.mockRejectedValue({
      response: { data: { detail: 'The selected scene has no valid model pixels.' } },
    });
    render(<AIInferencePanel sceneId="scene-3" />);
    fireEvent.click(await screen.findByRole('button', { name: 'Run AI land-cover inference' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The selected scene has no valid model pixels.',
    );
    expect(screen.queryByRole('region', { name: 'AI inference result' })).not.toBeInTheDocument();
  });

  it('shows model-registry request errors and provides no false ready state', async () => {
    getAIModels.mockRejectedValue({
      response: { data: { detail: 'Model registry is unavailable.' } },
    });
    render(<AIInferencePanel sceneId="scene-4" />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Model registry is unavailable.');
    expect(screen.getByRole('button', { name: 'Run AI land-cover inference' })).toBeDisabled();
    expect(getAIInferenceResult).not.toHaveBeenCalled();
  });
});
