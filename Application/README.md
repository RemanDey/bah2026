# SH-WFS Pipeline

Shack–Hartmann Wavefront Sensor — interactive diagnostic dashboard.

## Overview

Real-time SH-WFS simulation and processing pipeline with a modern web frontend. Generates synthetic subaperture images, estimates wavefront slopes, and reconstructs phase using multiple reconstructor algorithms.
<img width="1791" height="965" alt="app_screenshot" src="https://github.com/user-attachments/assets/7e870b82-bf2e-43e9-9a44-ec026a0b204a" />

<img width="1791" height="965" alt="app_screenshot2" src="https://github.com/user-attachments/assets/b44cdffd-5b83-44de-b314-bdf0ab5c08f5" />
<img width="1791" height="965" alt="image" src="https://github.com/user-attachments/assets/2d446980-55a0-4c38-9c5d-62e2f69323d8" />




## Features

- **Atmospheric turbulence simulation** — Kolmogorov phase screen or synthetic sinusoidal spot displacements
- **Centroid estimation** — Windowed center-of-mass or 2D Gaussian fitting via `scipy.optimize.curve_fit`
- **Phase reconstruction** — Three reconstructors: least-squares pseudo-inverse, Tikhonov-regularized (minimum variance), PINN-style nonlinear correction
- **Closed-loop control** — Configurable gain servo with residual feedback tracking
- **PSF computation** — Far-field diffraction pattern via FFT with pupil masking
- **Zernike decomposition** — 10-mode modal analysis (tip, tilt, defocus, astigmatism, coma, trefoil, spherical)
- **Real-time plotting** — Interactive Plotly charts: phase/actuator maps, PSF, Zernike bars, multi-metric trend, 3D phase surface, Strehl vs r₀ scatter
- **Analysis tab** — Slope vector bars, histogram, covariance heatmaps (32×32 full + 16×16 total-power), X-vs-Y slope scatter, temporal PSD, Zernike mode gallery, statistical summary
- **Kolmogorov phase screen** — FFT-based turbulent screen with temporal evolution
- **Record & playback** — Capture frame sequences and replay
- **Keyboard shortcuts** — Space (toggle auto-run), R (single frame), P (presets)
- **Preset system** — Light / Moderate / Severe / Extreme / Default turbulence profiles

## Requirements

- Python 3.10+
- Flask
- NumPy
- SciPy
- Gunicorn (optional, for production)

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in a browser.

### Production

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Architecture

```
app.py              — Flask backend (REST API, simulation engine)
templates/
  base.html         — Shell layout, Plotly CDN, tab navigation
  index.html        — Main dashboard + analysis tab UI
static/
  css/style.css     — Glassmorphism dark theme, responsive layout
```

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the dashboard HTML |
| `GET /api/step` | Run one pipeline frame (all params as query args) |
| `GET /api/advanced` | Slope vectors, histogram, statistics |
| `GET /api/covariance` | 32×32 + 16×16 slope covariance matrices |
| `GET /api/psd` | Temporal power spectral density (log-log) |
| `GET /api/zernike_modes` | Mode shapes + current coefficients |
| `GET /api/scatter` | Accumulated Strehl vs r₀ pairs |
| `GET /api/system` | Python, NumPy, SciPy versions, platform info |
| `GET /api/presets` | Turbulence preset definitions |

### Pipeline Stages

1. **Atmospheric phase** — Kolmogorov FFT screen or synthetic shifts injected as spot displacements on a 4×4 lenslet grid (64×64 px detector)
2. **Centroid extraction** — Per-subaperture gradient estimation via windowed COM or 2D Gaussian fit
3. **Phase reconstruction** — Linear reconstructor matrix mapping 32 slopes to 25 actuator commands
4. **DM control** — Pseudo-inverse decoupling matrix + nonlinear PINN edge correction
5. **PSF formation** — Far-field diffraction from reconstructed phase via FFT
6. **Modal analysis** — Zernike polynomial decomposition

## Configuration

All pipeline parameters are adjustable from the UI: turbulence severity, readout noise, photon noise, centroid method, loop gain, auto-run interval, Kolmogorov toggle.

