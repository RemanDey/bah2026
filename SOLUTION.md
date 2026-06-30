
## Advanced Real-Time Wavefront Reconstruction, Turbulence Profiling, and Predictive Control Pipeline for Shack-Hartmann WFS

This document describes a research-grade, hybrid AO (adaptive optics) processing stack for Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data. It combines an optimized classical BLAS/SIMD C-engine, GPU-accelerated reconstruction, a Kalman/LQG predictive controller, an asynchronous Physics-Informed Neural Network (PINN) actuator corrector, a single-frame deep-learning turbulence estimator, and a Cn²(h) altitude-resolved profiler — all under a strict end-to-end **≤ 10 ms** real-time deadline with deterministic worst-case latency.

---

### 0. System Overview & Error Budget

A closed-loop AO system's residual wavefront error decomposes as:

$$\sigma_{\text{total}}^2 = \sigma_{\text{fitting}}^2 + \sigma_{\text{servo-lag}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{aliasing}}^2 + \sigma_{\text{calibration}}^2$$

Each pipeline stage below is mapped to the term(s) it targets, so improvements are traceable to Strehl ratio gains rather than treated as isolated engineering wins:

| Stage | Targets | Mechanism |
|---|---|---|
| WCoM + dynamic threshold | $\sigma_{\text{noise}}$ | suppresses read/background noise in centroiding |
| Zonal/Modal reconstruction | $\sigma_{\text{fitting}}$ | SVD-regularized geometry matching |
| Kalman/LQG predictor | $\sigma_{\text{servo-lag}}$ | predicts ahead of frame-to-actuation delay |
| PINN residual corrector | $\sigma_{\text{calibration}}$ | corrects mirror non-linearity beyond linear $H^{-1}$ |
| Cn²(h) profiler | informs $\sigma_{\text{fitting}}$ budget | guides modal order / MCAO conjugate altitudes |
| Vibration/anomaly monitor | $\sigma_{\text{servo-lag}}$, safety | flags telescope/mount resonances in real time |

---

### 1. High-Speed Sub-Pixel Centroiding & Data I/O

#### 1.1 Pre-Cached, Page-Locked Parallel Loader
The full time-series is ingested into RAM at initialization. On GPU-accelerated builds, frames are additionally staged into **CUDA pinned (page-locked) host memory**, enabling asynchronous `cudaMemcpyAsync` transfers that overlap with CPU centroiding of the previous frame (double-buffered pipeline, Section 5.3).

#### 1.2 Weighted Center of Mass (WCoM) with Adaptive Windowing
A regional floor filter followed by spatial Gaussian weighting $W(x,y)$ suppresses read noise and edge clipping:

$$I_{\text{th}}(x,y) = \max\left(0, I(x,y) - I_{\text{bg}} - k \cdot \sigma_{\text{bg}}\right)$$

$$x_c = \frac{\sum_{x,y} x \cdot I_{\text{th}} \cdot W}{\sum_{x,y} I_{\text{th}} \cdot W}, \quad y_c = \frac{\sum_{x,y} y \cdot I_{\text{th}} \cdot W}{\sum_{x,y} I_{\text{th}} \cdot W}$$

**Enhancement — adaptive window radius:** the Gaussian weighting width $\sigma_W$ is no longer fixed. It is modulated per-subaperture each frame using the previous frame's spot FWHM (tracked via a running EMA), so windows tighten under good seeing (reducing noise propagation) and widen under strong scintillation (avoiding spot truncation):

$$\sigma_W^{(t)} = \text{clip}\left(\alpha \cdot \text{FWHM}^{(t-1)}_{\text{EMA}},\ \sigma_{\min},\ \sigma_{\max}\right)$$

**Enhancement — correlation-based centroiding fallback:** for low-flux subapertures (guide-star-limited regimes), the engine switches from WCoM to a normalized cross-correlation against a stored reference spot kernel, which is more robust at low SNR at the cost of extra FLOPs — applied only to flagged subapertures, not the full array, to preserve the timing budget.

#### 1.3 Bad/Saturated Subaperture Masking
Each frame, subapertures are flagged bad if (a) total flux falls below a noise-floor multiple, (b) the peak pixel saturates the ADC, or (c) the WCoM solution diverges outside the search box. Flagged slopes are not zeroed (which biases the reconstructor) but **excluded from $\mathbf{s}$ via a row-masked reconstruction matrix**, recomputed at low rate (~1 Hz) from the current bad-subaperture mask using a precomputed pseudo-inverse bank, avoiding a full SVD in the real-time path.

#### 1.4 Local Wavefront Gradients
$$s_x = \frac{x_c - x_{\text{ref}}}{f_{\text{MLA}}}, \quad s_y = \frac{y_c - y_{\text{ref}}}{f_{\text{MLA}}}, \quad \mathbf{s} \in \mathbb{R}^{2N\times1}$$

---

### 2. Reconstructive Architectures (Classical Core)

#### 2.1 Modal & Zonal Reconstruction
Offline SVD-regularized pseudo-inverses $R_{\text{modal}}, R_{\text{zonal}}$ give:

$$\boldsymbol{\phi} = R_{\text{zonal}} \cdot \mathbf{s}, \qquad O(N^2)$$

**Enhancement — Tikhonov-regularized truncated SVD with condition monitoring:** singular values below a noise-informed threshold $\epsilon = \kappa \cdot \sigma_{\text{noise}}$ are truncated at matrix-build time, and the resulting condition number is logged; if telemetry shows it drifting (e.g., due to dead actuators), the system flags a recalibration event rather than silently degrading.

#### 2.2 Matrix-Free Conjugate-Gradient Path for High-Order Systems
For very high actuator counts ($N > 2000$, e.g. ELT-scale), explicit $O(N^2)$ GEMV becomes the bottleneck. An optional **matrix-free preconditioned conjugate-gradient (PCG)** solver is provided, using a Fourier-domain (FFT-based) preconditioner for the Poisson-like reconstruction problem, reducing complexity toward $O(N \log N)$ per iteration with 3–5 iterations sufficient for convergence within the noise floor.

---

### 3. Predictive Control: Kalman Filter / LQG Layer

Pure proportional-integral control on reconstructed slopes incurs servo-lag error proportional to loop delay and turbulence temporal bandwidth. We add a predictive layer:

#### 3.1 State-Space Turbulence Model
Each tracked modal coefficient $a_i(t)$ is modeled as a damped, driven harmonic / AR(2) process (standard "Kalman AO" formulation):

$$\mathbf{x}_{k+1} = A\,\mathbf{x}_k + \mathbf{w}_k, \qquad \mathbf{z}_k = C\,\mathbf{x}_k + \mathbf{v}_k$$

where $\mathbf{x}_k$ stacks modal amplitude and velocity, $A$ encodes per-mode autoregressive coefficients fit offline from closed-loop telemetry PSDs, and $\mathbf{z}_k$ is the noisy slope-derived modal measurement.

#### 3.2 Steady-State Kalman Gain (LQG)
The steady-state gain $K_\infty$ solving the discrete algebraic Riccati equation is precomputed offline per mode (banded, since modes are largely decoupled in the modal basis), so the real-time update is a cheap per-mode 2×2 (or small block) operation:

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_\infty\left(\mathbf{z}_k - C\hat{\mathbf{x}}_{k|k-1}\right)$$
$$\hat{\mathbf{x}}_{k+1|k} = A\,\hat{\mathbf{x}}_{k|k}$$

The one-step-ahead prediction $\hat{\mathbf{x}}_{k+1|k}$ is what's actually commanded to the mirror, compensating for the frame-grab → reconstruction → DM-write latency.

#### 3.3 Vibration Rejection
Narrowband mechanical resonances (telescope mount, cryocooler) are detected via online Welch PSD estimation on the tip-tilt residual stream; detected peaks above a threshold automatically spawn additional AR poles in $A$ for those modes ("peak filtering"), giving targeted rejection without inflating gain across the whole bandwidth.

---

### 4. Statistical Turbulence Characterization

#### 4.1 Analytical Module (Sliding Temporal Queue)
$$r_0 = D_{\text{pupil}} \cdot \left(\frac{0.896}{\sigma_{\text{tilt}}^2}\right)^{3/5}$$

$\tau_0$ from the $1/e$ point of the normalized temporal autocorrelation of tip/tilt.

#### 4.2 Data-Driven Module (Single-Frame Neural Inference)
A lightweight CNN (ResNet-18-class) infers instantaneous $r_0$ from a single SH-WFS frame, trained with angular-even Zernike coefficients mapped to absolute values to avoid phase-sign ambiguity, running open- or closed-loop.

#### 4.3 New — Altitude-Resolved Cn²(h) Profiling (SLODAR-style)
Using the spatial cross-correlation of slopes between widely-separated guide directions (or, in single-guide-star mode, the temporal cross-correlation of slopes from sub-aperture pairs exploiting wind triangulation — "Generalized SLODAR"), the pipeline estimates a coarse turbulence profile $C_n^2(h)$ over a handful of altitude bins. This is used to:
- inform whether a single-conjugate correction is adequate or anisoplanatism is significant,
- set modal order / control bandwidth adaptively, and
- feed altitude priors to a future multi-conjugate AO (MCAO) extension.

#### 4.4 New — Transformer-Based Short-Horizon Wind/Turbulence Forecasting
A small causal transformer (operating on the same async thread as the CNN, time-shared) ingests the last ~50 frames of modal coefficients and outputs a short-horizon (2–5 frame) forecast of dominant turbulence layer velocity, used to seed/re-tune the Kalman model's $A$ matrix online rather than relying solely on the fixed offline fit — improving tracking through rapidly evolving seeing.

---

### 5. Actuator Map Inversion: Linear Baseline + PINN Residual + Constraints

#### 5.1 Linear Baseline
$$\mathbf{u}_{\text{linear}} = H^{-1} \cdot \mathbf{A}_{\text{target}}$$

#### 5.2 PINN Residual Corrector
Trained offline with a physics-guided loss embedding the biharmonic plate equation of the faceplate:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1\left\|\frac{\partial\hat\phi}{\partial x} - s_x\right\|_2^2 + \lambda_1\left\|\frac{\partial\hat\phi}{\partial y} - s_y\right\|_2^2 + \lambda_2\,\mathcal{L}_{\text{hysteresis}}$$

**Enhancement — hysteresis term:** $\mathcal{L}_{\text{hysteresis}}$ penalizes deviation from a Preisach-type hysteresis model fit to bench measurements of actuator response, so the PINN captures path-dependent (not just instantaneous) non-linearity — important for piezo-stack DMs.

Real-time inference runs with frozen weights under `torch.no_grad()` (or as a quantized INT8 TensorRT engine for lower latency):

$$\mathbf{u}_{\text{final}} = \text{clip}\left(\mathbf{u}_{\text{linear}} + \mathbf{u}_{\text{PINN\_residual}},\ \mathbf{u}_{\min},\ \mathbf{u}_{\max}\right)$$

#### 5.3 New — Hard Safety Layer (Stroke & Rate Limiting + Anomaly Veto)
Independent of the PINN's own out-of-bounds fallback, a deterministic C-side safety layer enforces, per actuator: (a) absolute stroke limits, (b) maximum slew rate between frames, and (c) a simple autoencoder-based anomaly score on the full command vector — if the score exceeds a threshold (suggesting a corrupted reconstruction, e.g. from a tracking-camera glitch), the system holds the last validated safe command rather than applying a potentially destructive one. This runs *after* the PINN, so it is the final authority regardless of upstream confidence.

---

### 6. GPU-Accelerated Variant (cuBLAS / CUDA Streams)

For high-actuator-count or high-frame-rate systems, the GEMV-bound steps are offloaded to GPU with a double-buffered, stream-pipelined design so frame $k$'s GPU reconstruction overlaps frame $k{+}1$'s CPU centroiding:

```c
// Pseudocode: double-buffered async pipeline (host orchestration)
cudaStream_t stream_recon, stream_xfer;

for (int k = 0; k < total_frames; ++k) {
    int cur = k % 2, prev = (k + 1) % 2;

    // Async H2D copy of next frame's centroiding inputs (pinned memory)
    cudaMemcpyAsync(d_slopes[cur], h_slopes[cur], slopes_bytes,
                     cudaMemcpyHostToDevice, stream_xfer);

    // GPU reconstruction for current frame on its own stream
    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha,
                d_R_matrix, M, d_slopes[cur], 1, &beta, d_phase[cur], 1);

    // CPU does WCoM centroiding for frame k+1 concurrently
    #pragma omp parallel for
    for (int i = 0; i < num_subapertures; i++) {
        compute_wcom(&ctx, k + 1, i, h_slopes[prev]);
    }

    cudaStreamSynchronize(stream_recon);
    dispatch_to_dm(d_phase[cur]);
}
```

Worst-case latency is bounded by profiling each stage on target hardware (CUDA events) and verifying the 99.9th-percentile frame time against the 10 ms budget — not just the mean, since AO control stability depends on bounded *worst-case* jitter.

---

### 7. Highly Optimized CPU Control Engine (C Implementation)

Improvements over the baseline: 64-byte cache-line alignment for AVX2/AVX-512 SIMD GEMV fallback, explicit restrict pointers, masked reconstruction for bad subapertures, integrated Kalman predictor, and a hard safety-clamp stage.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>   // AVX2/AVX-512 intrinsics fallback path

#define ALIGN64 __attribute__((aligned(64)))

typedef struct {
    int num_subapertures;
    int num_actuators;
    int total_frames;

    unsigned char **cached_bmp_frames;   // Pre-loaded, page-locked if GPU build

    double *R_matrix      ALIGN64;       // [num_actuators x 2*num_subapertures], bad-subap masked
    double *H_inv_matrix  ALIGN64;       // [num_actuators x num_actuators]
    double *kalman_A      ALIGN64;       // Per-mode block-diagonal state transition
    double *kalman_Kinf   ALIGN64;       // Precomputed steady-state Kalman gain

    double *slopes        ALIGN64;       // [2 * num_subapertures]
    double *phase_map     ALIGN64;       // [num_actuators]
    double *modal_state    ALIGN64;      // Kalman state estimate [2 * num_modes]
    double *dm_strokes_lin ALIGN64;
    double *dm_strokes_final ALIGN64;

    uint8_t *bad_subap_mask;             // Updated at ~1 Hz, not per-frame
    double   stroke_limit;
    double   rate_limit;
    double   prev_command[/* num_actuators */ 4096];
} AOControlPipeline;

/* Hard safety layer: stroke + rate limiting, applied last, unconditionally. */
static inline void apply_safety_clamp(AOControlPipeline *ctx, double *u) {
    #pragma omp simd
    for (int i = 0; i < ctx->num_actuators; i++) {
        double max_step = ctx->prev_command[i] + ctx->rate_limit;
        double min_step = ctx->prev_command[i] - ctx->rate_limit;
        double v = u[i];
        v = v > max_step ? max_step : (v < min_step ? min_step : v);
        v = v >  ctx->stroke_limit ?  ctx->stroke_limit : v;
        v = v < -ctx->stroke_limit ? -ctx->stroke_limit : v;
        u[i] = v;
        ctx->prev_command[i] = v;
    }
}

/**
 * Executes real-time phase reconstruction, predictive correction,
 * and actuator command decoupling. Bounded worst-case runtime: <= 10 ms.
 */
void ao_pipeline_execute_frame(AOControlPipeline *ctx, int frame_index) {

    // Step 0: Centroiding (OpenMP), skipping/flagging bad subapertures
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ctx->num_subapertures; i++) {
        if (ctx->bad_subap_mask[i]) continue;
        // ... WCoM with adaptive window + correlation fallback inserted here ...
    }

    // Step 1: Zonal reconstruction (bad-subap-masked R_matrix): BLAS GEMV
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->num_actuators, 2 * ctx->num_subapertures,
                1.0, ctx->R_matrix, 2 * ctx->num_subapertures,
                ctx->slopes, 1,
                0.0, ctx->phase_map, 1);

    // Step 2: Kalman predict+update on modal coefficients (small, per-mode)
    kalman_predict_update(ctx->kalman_A, ctx->kalman_Kinf,
                           ctx->modal_state, ctx->phase_map);

    // Step 3: Linear cross-talk decoupling against the *predicted* phase map
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->num_actuators, ctx->num_actuators,
                -1.0, ctx->H_inv_matrix, ctx->num_actuators,
                ctx->phase_map, 1,
                0.0, ctx->dm_strokes_lin, 1);

    // Step 4: PINN residual (TensorRT/ONNX C++ binding, frozen weights)
    get_pinn_residual(ctx->slopes, ctx->dm_strokes_final /* residual out */);
    #pragma omp simd
    for (int i = 0; i < ctx->num_actuators; i++)
        ctx->dm_strokes_final[i] += ctx->dm_strokes_lin[i];

    // Step 5: Anomaly check + hard stroke/rate safety clamp (final authority)
    if (autoencoder_anomaly_score(ctx->dm_strokes_final) > ANOMALY_THRESHOLD) {
        memcpy(ctx->dm_strokes_final, ctx->prev_command,
               sizeof(double) * ctx->num_actuators);
    } else {
        apply_safety_clamp(ctx, ctx->dm_strokes_final);
    }

    dispatch_to_dm(ctx->dm_strokes_final);
}
```

---

### 8. Calibration, Testing & Telemetry

- **Interaction-matrix calibration:** automated poke-matrix routine (push-pull per actuator) with outlier-robust fitting, run offline and on-demand; results feed both $H^{-1}$ and the bad-actuator mask.
- **Hardware-in-the-loop benchmarking:** worst-case (p99.9) and mean per-stage latency logged via high-resolution timers (`clock_gettime(CLOCK_MONOTONIC)` / CUDA events), regression-tested against the 10 ms budget on every build.
- **Unit/integration tests:** synthetic Kolmogorov phase screens (FFT or Zernike-sum method) pushed through the full pipeline to validate Strehl recovery and reconstruction matrix correctness against known ground truth.
- **Telemetry stream:** $r_0$, $\tau_0$, Cn²(h) bins, Kalman innovation residuals, anomaly scores, and per-actuator clamp-activation counts are logged at reduced rate for offline diagnosis and PINN/Kalman model retraining.

---

### 9. Summary of Additions Over Baseline

1. Adaptive WCoM window sizing + correlation-based low-SNR fallback.
2. Bad/saturated subaperture masking with low-rate reconstructor updates.
3. Truncated-SVD regularization with condition-number health monitoring.
4. Matrix-free PCG path for ELT-scale actuator counts.
5. Kalman/LQG predictive control layer with online vibration-peak rejection.
6. Cn²(h) altitude-resolved turbulence profiling (SLODAR-style).
7. Transformer-based short-horizon turbulence forecasting feeding the Kalman model.
8. Hysteresis-aware PINN loss for piezo-stack non-linearity.
9. Independent hard safety layer: stroke/rate limiting + autoencoder anomaly veto.
10. GPU/cuBLAS double-buffered streaming variant with worst-case latency profiling.
11. Calibration, HIL benchmarking, and synthetic-phase-screen test framework.
