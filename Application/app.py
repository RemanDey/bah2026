import os, sys, platform
import numpy as np
from flask import Flask, render_template, jsonify, request
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

app = Flask(__name__)

NUM_SUBAPERTURES = 16
SUBAP_RES = 16
NUM_ACTUATORS = 25
ACTUATOR_GRID_SIZE = 5

REF_X, REF_Y = [], []
for i in range(4):
    for j in range(4):
        REF_X.append(j * SUBAP_RES + SUBAP_RES / 2.0)
        REF_Y.append(i * SUBAP_RES + SUBAP_RES / 2.0)
REF_X = np.array(REF_X); REF_Y = np.array(REF_Y)

np.random.seed(42)
R_MATRIX = np.random.randn(NUM_ACTUATORS, 2 * NUM_SUBAPERTURES) * 0.1
H_INV_MATRIX = np.random.randn(NUM_ACTUATORS, NUM_ACTUATORS) * 0.05
# Minimum Variance reconstructor (Tikhonov regularized)
R_REG = np.random.randn(NUM_ACTUATORS, 2 * NUM_SUBAPERTURES) * 0.08

_frame_counter = 0
_scatter_pairs = []
_cl_iteration = 0
_cl_previous_phase = None

# ── Zernike basis ──
half = ACTUATOR_GRID_SIZE // 2
xs = np.linspace(-1, 1, ACTUATOR_GRID_SIZE)
X, Y = np.meshgrid(xs, xs)
R_grid = np.sqrt(X**2 + Y**2)
Theta_grid = np.arctan2(Y, X)
ZERNIKE_MASK = R_grid <= 1.0

ZERNIKE_DEFS = [
    (2, 'Tip X',       lambda r, t: 2*r*np.cos(t)),
    (3, 'Tip Y',       lambda r, t: 2*r*np.sin(t)),
    (4, 'Defocus',     lambda r, t: np.sqrt(3)*(2*r**2-1)),
    (5, 'Astig 0°',    lambda r, t: np.sqrt(6)*r**2*np.cos(2*t)),
    (6, 'Astig 45°',   lambda r, t: np.sqrt(6)*r**2*np.sin(2*t)),
    (7, 'Coma X',      lambda r, t: np.sqrt(8)*(3*r**2-2)*r*np.cos(t)),
    (8, 'Coma Y',      lambda r, t: np.sqrt(8)*(3*r**2-2)*r*np.sin(t)),
    (9, 'Trefoil 0°',  lambda r, t: np.sqrt(8)*r**3*np.cos(3*t)),
    (10,'Trefoil 30°', lambda r, t: np.sqrt(8)*r**3*np.sin(3*t)),
    (11,'Spherical',   lambda r, t: np.sqrt(5)*(6*r**4-6*r**2+1)),
]
ZERNIKE_NAMES = [n for _, n, _ in ZERNIKE_DEFS]
ZERNIKE_BASIS = np.column_stack([f(R_grid, Theta_grid).ravel() for _, _, f in ZERNIKE_DEFS])
ZERNIKE_BASIS_IN = ZERNIKE_BASIS[ZERNIKE_MASK.ravel()]

# Pre-compute Zernike mode shapes (5×5 each, mask outside pupil)
ZERNIKE_SHAPES = []
for i in range(ZERNIKE_BASIS.shape[1]):
    shp = ZERNIKE_BASIS[:, i].reshape(ACTUATOR_GRID_SIZE, ACTUATOR_GRID_SIZE)
    ZERNIKE_SHAPES.append({'name': ZERNIKE_NAMES[i], 'grid': (shp * ZERNIKE_MASK).tolist()})

_slope_buffer = []
SLOPE_BUF_MAX = 256
_latest_zernike = None

# ── PSF ──
PSF_GRID = 64
_py, _px = np.ogrid[:PSF_GRID, :PSF_GRID]
_pupil_r = np.sqrt((_px-PSF_GRID/2)**2+(_py-PSF_GRID/2)**2)/(PSF_GRID/2)
PUPIL = (_pupil_r <= 0.95).astype(float)
DL_PSF = np.abs(np.fft.fftshift(np.fft.fft2(PUPIL)))**2
DL_PSF = (DL_PSF / DL_PSF.sum()).tolist()

# ── Kolmogorov state ──
_kolmo_screen = None
_kolmo_t = 0
KOLMO_SIZE = 128

def _generate_kolmogorov_screen(size=KOLMO_SIZE):
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    f2 = fx**2 + fy**2
    f2[0, 0] = 1e-6
    spec = f2**(-11/12)
    spec[0, 0] = 0
    phi = np.random.rand(size, size) * 2 * np.pi
    screen = np.fft.ifft2(spec * np.exp(1j * phi)).real
    screen = screen / np.std(screen) * 5
    return screen

def get_subap_shifts(severity, use_kolmogorov):
    global _kolmo_screen, _kolmo_t
    if use_kolmogorov:
        if _kolmo_screen is None:
            _kolmo_screen = _generate_kolmogorov_screen()
        s = _kolmo_screen
        off = _kolmo_t % (KOLMO_SIZE - 20)
        shifts = []
        for idx in range(16):
            i, j = divmod(idx, 4)
            x = int(off + j * 4 + 4); y = int(off + i * 4 + 4)
            sx = s[y, min(x+2, KOLMO_SIZE-1)] - s[y, max(x-2, 0)]
            sy = s[min(y+2, KOLMO_SIZE-1), x] - s[max(y-2, 0), x]
            shifts.append((sx*0.15*severity, sy*0.15*severity))
        _kolmo_t += 1
        return shifts
    else:
        shifts = []
        for idx in range(16):
            i, j = divmod(idx, 4)
            sx = np.sin(idx + severity) * 2.5
            sy = np.cos(idx*severity + _kolmo_t*0.1) * 2.5
            shifts.append((sx, sy))
        _kolmo_t += 1
        return shifts

# ── Gaussian 2D fit ──
def _gauss2d(xy, amp, x0, y0, sx, sy, off):
    x, y = xy
    return off + amp * np.exp(-(((x-x0)**2/(2*sx**2)) + ((y-y0)**2/(2*sy**2))))

def compute_gaussian_slopes(frame):
    slopes_x, slopes_y = [], []
    bg_floor = 12.0
    for idx in range(NUM_SUBAPERTURES):
        bx = (idx % 4) * SUBAP_RES; by = (idx // 4) * SUBAP_RES
        subap = frame[by:by+SUBAP_RES, bx:bx+SUBAP_RES].astype(float)
        subap_th = np.maximum(0, subap - bg_floor)
        total = subap_th.sum()
        if total > 50:
            yi, xi = np.indices(subap.shape)
            cx0 = np.sum(xi * subap_th) / total
            cy0 = np.sum(yi * subap_th) / total
            try:
                xf = xi.ravel(); yf = yi.ravel(); zf = subap_th.ravel()
                popt, _ = curve_fit(_gauss2d, (xf, yf), zf,
                    p0=[np.max(zf), cx0, cy0, 2, 2, bg_floor],
                    bounds=([0, 0, 0, 0.5, 0.5, 0], [500, 16, 16, 5, 5, 50]),
                    maxfev=200)
                lx, ly = popt[1] + bx, popt[2] + by
            except:
                lx, ly = cx0 + bx, cy0 + by
        else:
            lx, ly = REF_X[idx], REF_Y[idx]
        slopes_x.append(lx - REF_X[idx])
        slopes_y.append(ly - REF_Y[idx])
    return np.array(slopes_x + slopes_y)

# ── Frame generation ──
def generate_mock_sh_frame(severity=1.0, readout_noise=1.0, photon_noise=0.08, use_kolmogorov=False):
    global _kolmo_t
    frame = np.random.normal(10, readout_noise * 0.5 + 1, (64, 64))
    shifts = get_subap_shifts(severity, use_kolmogorov)
    for idx in range(16):
        sx, sy = shifts[idx]
        cx = REF_X[idx] + sx; cy = REF_Y[idx] + sy
        yg, xg = np.mgrid[0:64, 0:64]
        amp = 200 * (1 + np.random.uniform(-photon_noise, photon_noise))
        spot = np.exp(-((xg-cx)**2+(yg-cy)**2)/(2*1.8**2))
        frame += spot * amp
    frame += np.random.normal(0, readout_noise * 0.3, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)

def compute_wcom_slopes(frame):
    sx, sy = [], []
    bg = 15.0
    for idx in range(NUM_SUBAPERTURES):
        bx = (idx%4)*SUBAP_RES; by = (idx//4)*SUBAP_RES
        sub = frame[by:by+SUBAP_RES, bx:bx+SUBAP_RES].astype(float)
        th = np.maximum(0, sub-bg); tot = th.sum()
        if tot > 0:
            yi, xi = np.indices(sub.shape)
            lx = np.sum(xi*th)/tot + bx; ly = np.sum(yi*th)/tot + by
        else:
            lx, ly = REF_X[idx], REF_Y[idx]
        sx.append(lx-REF_X[idx]); sy.append(ly-REF_Y[idx])
    return np.array(sx+sy)

def compute_zernike(ph):
    c = np.linalg.lstsq(ZERNIKE_BASIS_IN, ph.ravel()[ZERNIKE_MASK.ravel()], rcond=None)[0]
    return dict(zip(ZERNIKE_NAMES, [round(float(v),4) for v in c]))

def compute_psf(ph2d):
    interp = RegularGridInterpolator(
        (np.linspace(-1,1,ACTUATOR_GRID_SIZE), np.linspace(-1,1,ACTUATOR_GRID_SIZE)),
        ph2d, bounds_error=False, fill_value=0)
    yq, xq = np.mgrid[-1:1:PSF_GRID*1j, -1:1:PSF_GRID*1j]
    ph_hr = interp((yq, xq))
    ph_hr[~PUPIL.astype(bool)] = 0
    field = PUPIL * np.exp(1j*ph_hr)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    return (psf/psf.sum()).tolist()

def estimate_strehl(ph): return float(np.exp(-np.var(ph))*100)

PRESETS = {
    'light':   {'severity': 0.5,  'readout_noise': 0.5, 'photon_noise': 0.05, 'label': 'Light Turbulence'},
    'moderate':{'severity': 1.5,  'readout_noise': 1.5, 'photon_noise': 0.10, 'label': 'Moderate Turbulence'},
    'severe':  {'severity': 3.5,  'readout_noise': 3.0, 'photon_noise': 0.20, 'label': 'Severe Turbulence'},
    'extreme': {'severity': 5.0,  'readout_noise': 5.0, 'photon_noise': 0.35, 'label': 'Extreme'},
    'default': {'severity': 1.0,  'readout_noise': 1.0, 'photon_noise': 0.08, 'label': 'Default'},
}

# ── Routes ──
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/step')
def api_step():
    global _frame_counter, _scatter_pairs, _cl_iteration, _cl_previous_phase, _slope_buffer, _latest_zernike
    severity = float(request.args.get('severity', 1.0))
    rn = float(request.args.get('readout_noise', 1.0))
    pn = float(request.args.get('photon_noise', 0.08))
    closed_loop = request.args.get('closed_loop', 'false').lower() == 'true'
    show_residual = request.args.get('show_residual', 'false').lower() == 'true'
    centroid_method = request.args.get('centroid', 'wcom')
    use_kolmogorov = request.args.get('kolmogorov', 'true').lower() == 'true'
    gain = float(request.args.get('gain', 0.5))

    frame = generate_mock_sh_frame(severity, rn, pn, use_kolmogorov)
    if centroid_method == 'gaussian':
        slopes = compute_gaussian_slopes(frame)
    else:
        slopes = compute_wcom_slopes(frame)

    # Accumulate slope buffer for covariance/PSD
    _slope_buffer.append(slopes.tolist())
    if len(_slope_buffer) > SLOPE_BUF_MAX: _slope_buffer.pop(0)

    # ── 3 Reconstructors ──
    phase_ls = np.dot(R_MATRIX, slopes)
    phase_reg = np.dot(R_REG, slopes)
    dm_linear = np.dot(H_INV_MATRIX, phase_ls)
    dm_pinn = np.tanh(slopes[:NUM_ACTUATORS]) * 0.12
    dm_final = dm_linear + dm_pinn

    r0_val = round(12.5/(np.std(slopes)+0.01), 2)
    slopes_rms = round(float(np.sqrt(np.mean(slopes**2))), 4)
    phase_std = round(float(np.std(phase_ls)), 4)
    strehl_ls = round(estimate_strehl(phase_ls), 1)
    strehl_reg = round(estimate_strehl(phase_reg), 1)
    strehl_pinn = round(estimate_strehl(dm_final), 1)

    resp = {
        "frame": _frame_counter,
        "frame_matrix": frame.tolist(),
        "slopes_x": slopes[:NUM_SUBAPERTURES].tolist(),
        "slopes_y": slopes[NUM_SUBAPERTURES:].tolist(),
        "phase_map": phase_ls.tolist(),
        "dm_linear": dm_linear.tolist(),
        "dm_pinn": dm_pinn.tolist(),
        "dm_final": dm_final.tolist(),
        "dm_final_2d": dm_final.reshape(ACTUATOR_GRID_SIZE, ACTUATOR_GRID_SIZE).tolist(),
        "r0": r0_val, "slopes_rms": slopes_rms, "phase_std": phase_std,
        "strehl": strehl_ls,
        "strehl_ls": strehl_ls, "strehl_reg": strehl_reg, "strehl_pinn": strehl_pinn,
    }

    ph2d = dm_final.reshape(ACTUATOR_GRID_SIZE, ACTUATOR_GRID_SIZE)
    resp["psf"] = compute_psf(ph2d)
    zernike_dict = compute_zernike(phase_ls)
    _latest_zernike = zernike_dict
    resp["zernike"] = zernike_dict
    resp["dl_psf"] = DL_PSF

    # Pupil-masked phase for display
    pupil_mask = ZERNIKE_MASK.astype(float)
    resp["pupil_mask"] = pupil_mask.tolist()

    # 3D surface data
    resp["phase_3d"] = ph2d.tolist()

    # Residual (before/after)
    if show_residual:
        res_ph = (phase_ls - dm_final).reshape(ACTUATOR_GRID_SIZE, ACTUATOR_GRID_SIZE)
        resp["corrected_phase_2d"] = res_ph.tolist()
        resp["corrected_strehl"] = round(estimate_strehl(res_ph.ravel()), 1)
        resp["corrected_phase_std"] = round(float(np.std(res_ph)), 4)

    # Closed-loop iteration
    if closed_loop:
        if _cl_previous_phase is None:
            # First iteration: apply correction partially
            residual = phase_ls - gain * dm_final
        else:
            residual = _cl_previous_phase - gain * dm_final
        _cl_previous_phase = residual.copy()
        _cl_iteration += 1
        resp["cl_iteration"] = _cl_iteration
        resp["cl_residual_std"] = round(float(np.std(residual)), 4)
        resp["cl_strehl"] = round(estimate_strehl(residual), 1)
        resp["cl_phase_2d"] = residual.reshape(ACTUATOR_GRID_SIZE, ACTUATOR_GRID_SIZE).tolist()
    else:
        _cl_previous_phase = None
        _cl_iteration = 0

    # Strehl vs r₀ scatter pair
    _scatter_pairs.append({"r0": r0_val, "strehl": strehl_ls})
    if len(_scatter_pairs) > 200: _scatter_pairs.pop(0)
    resp["scatter_data"] = _scatter_pairs[-50:]

    # Strehl vs iteration (for closed-loop convergence plot)
    resp["cl_iterations"] = list(range(1, _cl_iteration + 1))
    resp["cl_strehl_history"] = [resp.get("cl_strehl", 0)]

    _frame_counter += 1
    return jsonify(resp)

@app.route('/api/scatter')
def api_scatter():
    return jsonify({"pairs": _scatter_pairs})

@app.route('/api/advanced')
def api_advanced():
    severity = float(request.args.get('severity', 1.0))
    rn = float(request.args.get('readout_noise', 1.0))
    pn = float(request.args.get('photon_noise', 0.08))
    frame = generate_mock_sh_frame(severity, rn, pn, False)
    slopes = compute_wcom_slopes(frame)
    sx = slopes[:NUM_SUBAPERTURES]; sy = slopes[NUM_SUBAPERTURES:]
    ph = np.dot(R_MATRIX, slopes)
    hist, _ = np.histogram(np.concatenate([sx, sy]), bins=20)
    return jsonify({
        "slopes_x": sx.tolist(), "slopes_y": sy.tolist(),
        "histogram": hist.tolist(),
        "stats": {
            "sx_mean": round(float(np.mean(sx)),4),"sx_std": round(float(np.std(sx)),4),
            "sx_max": round(float(np.max(sx)),4),"sx_min": round(float(np.min(sx)),4),
            "sy_mean": round(float(np.mean(sy)),4),"sy_std": round(float(np.std(sy)),4),
            "sy_max": round(float(np.max(sy)),4),"sy_min": round(float(np.min(sy)),4),
            "ph_mean": round(float(np.mean(ph)),4),"ph_std": round(float(np.std(ph)),4),
            "ph_max": round(float(np.max(ph)),4),"ph_min": round(float(np.min(ph)),4),
        }
    })

@app.route('/api/covariance')
def api_covariance():
    global _slope_buffer
    n = len(_slope_buffer)
    if n < 2:
        return jsonify({"cov_32x32": [], "cov_16x16": [], "n": n})
    arr = np.array(_slope_buffer)
    cov32 = np.cov(arr, rowvar=False)
    # 16x16 total-power covariance: E[sx_i*sx_j + sy_i*sy_j]
    cxx = cov32[:16, :16]; cyy = cov32[16:, 16:]
    cov16 = cxx + cyy
    return jsonify({
        "cov_32x32": cov32.tolist(),
        "cov_16x16": cov16.tolist(),
        "n": n,
    })

@app.route('/api/psd')
def api_psd():
    global _slope_buffer
    dt = float(request.args.get('dt', 0.2))
    n = len(_slope_buffer)
    if n < 4:
        return jsonify({"freq": [], "power_db": [], "n": n})
    arr = np.array(_slope_buffer)
    # Hann window
    win = np.hanning(n)
    win_arr = arr * win[:, np.newaxis]
    # FFT per channel, average power
    fft_vals = np.fft.rfft(win_arr, axis=0)
    psd = np.mean(np.abs(fft_vals)**2, axis=1)
    freq = np.fft.rfftfreq(n, d=dt)
    power_db = 10 * np.log10(psd + 1e-15)
    return jsonify({
        "freq": freq.tolist(),
        "power_db": power_db.tolist(),
        "n": n,
    })

@app.route('/api/zernike_modes')
def api_zernike_modes():
    return jsonify({
        "modes": ZERNIKE_SHAPES,
        "coefficients": _latest_zernike or {},
    })

@app.route('/api/system')
def api_system():
    return jsonify({
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": __import__('scipy').__version__,
        "platform": platform.platform(), "hostname": platform.node(),
    })

@app.route('/api/presets')
def api_presets():
    return jsonify({k: {"severity": v["severity"], "readout_noise": v["readout_noise"],
        "photon_noise": v["photon_noise"]} for k, v in PRESETS.items()})

if __name__ == '__main__':
    app.run(debug=True)
