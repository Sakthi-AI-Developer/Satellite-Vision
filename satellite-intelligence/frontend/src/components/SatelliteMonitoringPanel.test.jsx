import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SatelliteMonitoringPanel from './SatelliteMonitoringPanel';
import * as service from '../services/api';

vi.mock('../services/api', () => ({
  downloadMonitoringArtifact: vi.fn(),
  getMonitoringStatus: vi.fn(),
  listMonitoringScenes: vi.fn(),
  requestMonitoringScan: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  service.getMonitoringStatus.mockResolvedValue({
    enabled: true,
    status: 'idle',
    timezone: 'Asia/Kolkata',
    schedule_time: '03:00',
    data_source: 'Copernicus Data Space Ecosystem',
    collection: 'sentinel-2-l2a',
    model: { status: 'model_not_ready' },
    scene_counts: { discovered: 1, downloaded: 0, rejected: 0, analyzed: 0, failed: 0 },
  });
  service.listMonitoringScenes.mockResolvedValue({
    scenes: [{
      scene_id: 'S2A_TEST_SCENE',
      status: 'model_not_ready',
      observation_at: '2026-01-02T03:00:00Z',
      published_at: '2026-01-03T03:00:00Z',
      discovered_at: '2026-01-04T03:00:00Z',
      cloud_cover: 8,
    }],
  });
  service.requestMonitoringScan.mockResolvedValue({ status: 'accepted' });
});

describe('SatelliteMonitoringPanel', () => {
  it('shows actual scene timestamps and does not imply inference ran without a model', async () => {
    render(<SatelliteMonitoringPanel />);
    expect(await screen.findByText('model_not_ready')).toBeInTheDocument();
    expect(screen.getByText(/Observation time:/)).toBeInTheDocument();
    expect(screen.getByText(/Catalog availability time:/)).toBeInTheDocument();
    expect(screen.getByText(/First discovered:/)).toBeInTheDocument();
    expect(screen.getByText('Status: model_not_ready · Cloud cover: 8%')).toBeInTheDocument();
    expect(screen.queryByText(/Download real classification raster/)).not.toBeInTheDocument();
  });

  it('starts a manual scan and refreshes persisted status', async () => {
    render(<SatelliteMonitoringPanel />);
    fireEvent.click(await screen.findByRole('button', { name: 'Scan now' }));
    await waitFor(() => expect(service.requestMonitoringScan).toHaveBeenCalledOnce());
    expect(service.getMonitoringStatus).toHaveBeenCalledTimes(2);
  });

  it('displays a backend error instead of fabricating monitor state', async () => {
    service.getMonitoringStatus.mockRejectedValueOnce({
      response: { data: { detail: 'Monitoring migration is not applied.' } },
    });
    render(<SatelliteMonitoringPanel />);
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Monitoring migration is not applied.',
    );
  });
});
