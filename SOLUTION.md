## Advanced Real-Time Wavefront Reconstruction, Turbulence Profiling, and Predictive Control Pipeline for Shack-Hartmann WFS, with Auto-PINN Calibration

This document describes a research-grade, hybrid AO (adaptive optics) processing stack for Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data. It combines an optimized classical BLAS/SIMD C-engine, GPU-accelerated reconstruction, a Kalman/LQG predictive controller, an **auto-tuned, self-optimizing Physics-Informed Neural Network (Auto-PINN) actuator corrector**, a single-frame deep-learning turbulence estimator, and a Cn²(h) altitude-resolved profiler — all under a strict end-to-end **≤ 10 ms** real-time deadline with deterministic worst-case latency.

---

### 0. System Overview & Error Budget

A closed-loop AO system's residual wavefront error decomposes as:

$$\sigma_{\text{total}}^2 = \sigma_{\text{fitting}}^2 + \sigma_{\text{servo-lag}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{aliasing}}^2 + \sigma_{\text{calibration}}^2 + \sigma_{\text{drift}}^2$$

A new term, $\sigma_{\text{drift}}^2$, is introduced explicitly: the slowly-varying component of residual error caused by **non-stationary actuator response** — thermal creep, piezo aging, faceplate fatigue — that a *fixed*, offline-trained PINN cannot track. This is the term Auto-PINN is designed to drive toward zero over the observing run, distinct from $\sigma_{\text{calibration}}^2$, which captures the *instantaneous* (but time-invariant) non-linearity a correctly-calibrated network already handles.

| Stage | Targets | Mechanism |
|---|---|---|
| WCoM + dynamic threshold | $\sigma_{\text{noise}}$ | suppresses read/background noise in centroiding |
| Zonal/Modal reconstruction | $\sigma_{\text{fitting}}$ | SVD-regularized geometry matching |
| Kalman/LQG predictor | $\sigma_{\text{servo-lag}}$ | predicts ahead of frame-to-actuation delay |
| PINN residual corrector (static) | $\sigma_{\text{calibration}}$ | corrects mirror non-linearity beyond linear $H^{-1}$ |
| **Auto-PINN controller** | $\sigma_{\text{drift}}$, $\sigma_{\text{calibration}}$ | online architecture/weight/hyperparameter adaptation, tracks non-stationary actuator physics |
| Cn²(h) profiler | informs $\sigma_{\text{fitting}}$ budget | guides modal order / MCAO conjugate altitudes |
| Vibration/anomaly monitor | $\sigma_{\text{servo-lag}}$, safety | flags telescope/mount resonances in real time |

The remainder of this document treats the classical pipeline (Sections 1–4, 6–8) largely as before, then expands Section 5 to formalize Auto-PINN as a closed-loop meta-controller sitting *around* the existing PINN residual corrector rather than replacing it.

---

### 1. High-Speed Sub-Pixel Centroiding & Data I/O

#### 1.1 Pre-Cached, Page-Locked Parallel Loader
The full time-series is ingested into RAM at initialization. On GPU-accelerated builds, frames are additionally staged into **CUDA pinned (page-locked) host memory**, enabling asynchronous `cudaMemcpyAsync` transfers that overlap with CPU centroiding of the previous frame (double-buffered pipeline, Section 6).

#### 1.2 Weighted Center of Mass (WCoM) with Adaptive Windowing
A regional floor filter followed by spatial Gaussian weighting $W(x,y)$ suppresses read noise and edge clipping:

$$I_{\text{th}}(x,y) = \max\left(0, I(x,y) - I_{\text{bg}} - k \cdot \sigma_{\text{bg}}\right)$$

$$x_c = \frac{\sum_{x,y} x \cdot I_{\text{th}} \cdot W}{\sum_{x,y} I_{\text{th}} \cdot W}, \quad y_c = \frac{\sum_{x,y} y \cdot I_{\text{th}} \cdot W}{\sum_{x,y} I_{\text{th}} \cdot W}$$

**Adaptive window radius:** the Gaussian weighting width $\sigma_W$ is modulated per-subaperture each frame using the previous frame's spot FWHM (tracked via a running EMA):

$$\sigma_W^{(t)} = \text{clip}\left(\alpha \cdot \text{FWHM}^{(t-1)}_{\text{EMA}},\ \sigma_{\min},\ \sigma_{\max}\right)$$

The Cramér–Rao bound for centroid estimation under photon and read noise,

$$\sigma_{\text{centroid}}^2 \gtrsim \frac{\sigma_{\text{spot}}^2}{\text{SNR}^2}\left(1 + \frac{4\sqrt{\pi}\,\sigma_{\text{spot}}\,\sigma_{\text{bg}}^2}{N_{\text{ph}}\,p}\right)$$

(with $p$ the pixel pitch, $N_{\text{ph}}$ the photon count) motivates the adaptive window: tightening $\sigma_W$ reduces the effective background-noise term at the cost of truncating wings under scintillation, so $\alpha$ is itself tuned offline against a Cramér–Rao-optimal operating curve rather than chosen heuristically.

**Correlation-based centroiding fallback:** for low-flux subapertures, the engine switches from WCoM to normalized cross-correlation against a stored reference spot kernel, applied only to flagged subapertures to preserve the timing budget.

#### 1.3 Bad/Saturated Subaperture Masking
Flagged slopes are excluded from $\mathbf{s}$ via a row-masked reconstruction matrix, recomputed at low rate (~1 Hz) from a precomputed pseudo-inverse bank, avoiding a full SVD in the real-time path.

#### 1.4 Local Wavefront Gradients
$$s_x = \frac{x_c - x_{\text{ref}}}{f_{\text{MLA}}}, \quad s_y = \frac{y_c - y_{\text{ref}}}{f_{\text{MLA}}}, \quad \mathbf{s} \in \mathbb{R}^{2N\times1}$$

---

### 2. Reconstructive Architectures (Classical Core)

#### 2.1 Modal & Zonal Reconstruction
Offline SVD-regularized pseudo-inverses $R_{\text{modal}}, R_{\text{zonal}}$ give:

$$\boldsymbol{\phi} = R_{\text{zonal}} \cdot \mathbf{s}, \qquad O(N^2)$$

**Tikhonov-regularized truncated SVD with condition monitoring:** singular values below a noise-informed threshold $\epsilon = \kappa \cdot \sigma_{\text{noise}}$ are truncated at matrix-build time. The condition number $\kappa(R) = \sigma_{\max}/\sigma_{\min}$ is logged continuously; sustained upward drift (e.g. from dead actuators) is exactly the kind of slow non-stationarity that also feeds the Auto-PINN retraining trigger (Section 5.4.3), so this telemetry channel is shared rather than duplicated.

#### 2.2 Matrix-Free Conjugate-Gradient Path for High-Order Systems
For $N > 2000$ (ELT-scale), an optional **matrix-free preconditioned conjugate-gradient (PCG)** solver uses an FFT-based preconditioner for the Poisson-like reconstruction problem, with the standard CG convergence bound

$$\frac{\|\mathbf{e}_k\|_A}{\|\mathbf{e}_0\|_A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{k}$$

giving 3–5 iterations sufficient once the FFT preconditioner reduces $\kappa$ to near-unity for the dominant Laplacian-like operator.

---

### 3. Predictive Control: Kalman Filter / LQG Layer

#### 3.1 State-Space Turbulence Model
Each tracked modal coefficient $a_i(t)$ is modeled as an AR(2) process:

$$\mathbf{x}_{k+1} = A\,\mathbf{x}_k + \mathbf{w}_k, \qquad \mathbf{z}_k = C\,\mathbf{x}_k + \mathbf{v}_k$$

with $A$ fit offline from closed-loop telemetry PSDs (consistent with the von Kármán temporal power spectrum $\propto f^{-8/3}$ in the inertial range, breaking to $f^{-17/3}$ beyond the Greenwood frequency).

#### 3.2 Steady-State Kalman Gain (LQG)
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_\infty\left(\mathbf{z}_k - C\hat{\mathbf{x}}_{k|k-1}\right), \qquad \hat{\mathbf{x}}_{k+1|k} = A\,\hat{\mathbf{x}}_{k|k}$$

$K_\infty$ solves the discrete algebraic Riccati equation offline, per mode, as a small block operation.

#### 3.3 Vibration Rejection
Narrowband mechanical resonances are detected via online Welch PSD estimation on tip-tilt residuals; detected peaks spawn additional AR poles in $A$ for those modes ("peak filtering").

---

### 4. Statistical Turbulence Characterization

#### 4.1 Analytical Module
$$r_0 = D_{\text{pupil}} \cdot \left(\frac{0.896}{\sigma_{\text{tilt}}^2}\right)^{3/5}$$

$\tau_0$ from the $1/e$ point of the normalized temporal autocorrelation of tip/tilt.

#### 4.2 Data-Driven Module
A lightweight CNN (ResNet-18-class) infers instantaneous $r_0$ from a single SH-WFS frame, trained with angular-even Zernike coefficients to avoid phase-sign ambiguity.

#### 4.3 Altitude-Resolved Cn²(h) Profiling (SLODAR-style)
Spatial or temporal slope cross-correlation between sub-aperture pairs (Generalized SLODAR with wind triangulation) estimates a coarse $C_n^2(h)$ over a handful of altitude bins, informing anisoplanatism assessment, adaptive modal order, and future MCAO conjugate-altitude priors.

#### 4.4 Transformer-Based Short-Horizon Wind/Turbulence Forecasting
A small causal transformer ingests the last ~50 frames of modal coefficients and forecasts 2–5 frames ahead of dominant layer velocity, used to re-tune the Kalman $A$ matrix online. This forecaster and the CNN of 4.2 time-share an asynchronous inference thread with the **Auto-PINN background worker** described below (Section 5.4.4), under a single latency-budgeted scheduler so none of the three ever competes with the hard real-time loop for cycles.

---

### 5. Actuator Map Inversion: Linear Baseline + Auto-PINN Residual + Constraints

#### 5.1 Linear Baseline
$$\mathbf{u}_{\text{linear}} = H^{-1} \cdot \mathbf{A}_{\text{target}}$$

#### 5.2 Static PINN Residual Corrector (Baseline Network)
Trained offline with a physics-guided loss embedding the biharmonic plate equation of the faceplate:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1\left\|\frac{\partial\hat\phi}{\partial x} - s_x\right\|_2^2 + \lambda_1\left\|\frac{\partial\hat\phi}{\partial y} - s_y\right\|_2^2 + \lambda_2\,\mathcal{L}_{\text{hysteresis}} + \lambda_3\,\mathcal{L}_{\text{PDE}}$$

where $\mathcal{L}_{\text{PDE}}$ is the residual of the biharmonic operator $D\nabla^4 w = q(x,y)$ evaluated at collocation points via automatic differentiation (the defining feature of a PINN: the governing PDE is enforced as a soft constraint, not just fit from data), and $\mathcal{L}_{\text{hysteresis}}$ penalizes deviation from a Preisach-type hysteresis model fit to bench measurements, capturing path-dependent piezo-stack non-linearity.

Real-time inference runs with frozen weights under `torch.no_grad()` or as a quantized INT8 TensorRT engine:

$$\mathbf{u}_{\text{final}} = \text{clip}\left(\mathbf{u}_{\text{linear}} + \mathbf{u}_{\text{PINN\_residual}},\ \mathbf{u}_{\min},\ \mathbf{u}_{\max}\right)$$

The limitation this section exists to solve: $\mathcal{L}_{\text{total}}$'s minimizer, $\lambda_1,\lambda_2,\lambda_3$, and the network's own architecture (depth, width, activation family) are all chosen **once**, offline, by a human. As the DM's physical response drifts over an observing run or campaign, this static optimum stops being optimal — this is precisely $\sigma_{\text{drift}}^2$ from Section 0.

#### 5.3 New — Auto-PINN: Automated, Self-Optimizing Physics-Informed Residual Corrector

Auto-PINN is the integration of **automated machine learning (AutoML) for physics-informed networks** into the actuator-correction pipeline: it treats the PINN's architecture, loss-term weighting, and (when triggered) its weights themselves as variables to be optimized continuously and automatically against live telemetry, rather than fixed offline hyperparameters. It is implemented as four cooperating subsystems, all of which run **outside** the hard 10 ms real-time path — only the resulting frozen, validated network is ever swapped into the inference slot used by Section 5.2.

**5.3.1 Latency-Constrained Neural Architecture Search (NAS)**

Network depth/width/activation choice is posed as a bi-level optimization over a differentiable architecture search space (DARTS-style mixed operations, relaxed with a softmax over candidate layers $\alpha$), constrained to respect the inference-time budget allocated to Step 4 of the real-time loop ($\lesssim 1$–$2$ ms on the target accelerator):

$$\min_{\alpha} \;\; \mathcal{L}_{\text{val}}\big(w^*(\alpha), \alpha\big) \quad \text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{total}}(w, \alpha), \quad T_{\text{infer}}(\alpha) \le T_{\text{budget}}$$

The latency constraint is folded into the search objective as a differentiable penalty, $\mathcal{L}_{\text{val}} + \mu \cdot \max(0,\, T_{\text{infer}}(\alpha) - T_{\text{budget}})^2$, using a pre-measured per-operation latency lookup table on the deployment hardware (so the search never proposes an architecture that would blow the AO error budget even before considering accuracy). This runs offline/periodically (e.g. between observing blocks), not per-frame.

**5.3.2 Automated Hyperparameter & Loss-Weight Optimization**

Given a fixed (or NAS-selected) architecture, $\lambda_1, \lambda_2, \lambda_3$, learning-rate schedule, and ensemble size are tuned via population-based training (PBT) or Bayesian optimization (Gaussian-process surrogate over the validation loss surface), using the same closed-loop telemetry (Section 8) as the objective signal rather than a static held-out set alone — so the loss weighting tracks which error term ($\sigma_{\text{fitting}}$ vs. $\sigma_{\text{calibration}}$ vs. hysteresis-driven error) currently dominates on-sky.

**5.3.3 Meta-Learning for Fast Online Adaptation (Few-Shot Recalibration)**

Rather than retraining from scratch when drift is detected, Auto-PINN maintains a meta-initialization $\theta_0$ trained via a Reptile/MAML-style objective across historical drift scenarios (thermal cycles, simulated dead-actuator patterns, aging curves), such that a small number of gradient steps on a short window of fresh poke-matrix or on-sky residual data recovers a near-optimal corrector:

$$\theta_0 \leftarrow \theta_0 - \eta \sum_{i} \nabla_{\theta_0}\, \mathcal{L}_i\big(\theta_i'\big), \qquad \theta_i' = \theta_0 - \eta' \nabla_{\theta_0}\mathcal{L}_i(\theta_0)$$

This converts what would be a multi-hour offline retrain into a sub-minute few-shot adaptation, run in the background while the previous network continues serving inference.

**5.3.4 Automated Retraining Trigger & Uncertainty Quantification**

A background scheduler monitors three telemetry channels already produced elsewhere in the pipeline — (a) the reconstruction-matrix condition number (Section 2.1), (b) the hard-safety-layer anomaly score (Section 5.4 below), and (c) a rolling estimate of $\sigma_{\text{calibration}}^2$ derived from Kalman innovation residuals (Section 3.2) — and fires meta-adaptation (5.3.3) when any channel crosses a statistically-derived (CUSUM or Page-Hinkley) drift threshold, or triggers a full NAS+HPO cycle (5.3.1–5.3.2) on a slower cadence (e.g. nightly, or on operator demand) when drift is large or persistent.

The deployed network is additionally an ensemble (deep ensemble or MC-dropout at fixed inference cost via cached dropout masks), so each residual correction ships with a calibrated epistemic-uncertainty estimate $\hat\sigma_{\text{PINN}}(\mathbf{s})$. This is *not* decorative: it is fed directly into the hard safety layer of Section 5.4, so the anomaly veto considers not just the magnitude of the proposed command but the network's own confidence in it — a high-magnitude, high-confidence correction is treated very differently from a high-magnitude, low-confidence one.

**5.3.5 Champion/Challenger Deployment**

A newly adapted or searched network is never hot-swapped directly into the real-time inference slot. It is first run shadow-mode (challenger) in parallel with the currently deployed network (champion) on live data, with its outputs logged but not commanded to the DM. Promotion to champion requires (i) a statistically significant reduction in rolling $\sigma_{\text{calibration}}^2$/$\sigma_{\text{drift}}^2$ over a minimum dwell window, and (ii) no increase in worst-case (p99.9) inference latency beyond $T_{\text{budget}}$, verified against the same CUDA-event/`clock_gettime` profiling used in Section 6. This keeps Auto-PINN's self-modification strictly outside the path that could ever destabilize the closed loop.

```
Auto-PINN background orchestration (conceptual, asynchronous):

  telemetry_monitor()             # condition number, anomaly score, innovation residuals
      -> drift_score

  if drift_score > theta_fast:
      challenger = meta_adapt(champion, recent_window)     # Sec 5.3.3, few-shot
  if drift_score > theta_slow or scheduled_nightly_cycle:
      challenger = nas_plus_hpo(historical_telemetry)       # Sec 5.3.1 + 5.3.2

  run_shadow_mode(challenger, live_stream, duration=T_dwell)
  if validate_promotion(challenger, champion):              # Sec 5.3.5
      champion = freeze_and_quantize(challenger)             # -> TensorRT/ONNX INT8
      hot_swap_inference_engine(champion)                    # atomic pointer swap, no frame drop
```

#### 5.4 Hard Safety Layer (Stroke & Rate Limiting + Anomaly Veto)

Independent of the PINN's own behavior, a deterministic C-side safety layer enforces, per actuator: (a) absolute stroke limits, (b) maximum slew rate between frames, and (c) an autoencoder-based anomaly score on the full command vector, **now additionally weighted by the Auto-PINN ensemble's epistemic uncertainty** $\hat\sigma_{\text{PINN}}$ from Section 5.3.4 — corrections that are both large and low-confidence are vetoed more aggressively than large, high-confidence ones. If the combined score exceeds threshold, the system holds the last validated safe command. This runs *after* the PINN, so it is the final authority regardless of upstream confidence or Auto-PINN's promotion status.

---

### 6. GPU-Accelerated Variant (cuBLAS / CUDA Streams)

```c
// Pseudocode: double-buffered async pipeline (host orchestration)
cudaStream_t stream_recon, stream_xfer;

for (int k = 0; k < total_frames; ++k) {
    int cur = k % 2, prev = (k + 1) % 2;

    cudaMemcpyAsync(d_slopes[cur], h_slopes[cur], slopes_bytes,
                     cudaMemcpyHostToDevice, stream_xfer);

    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha,
                d_R_matrix, M, d_slopes[cur], 1, &beta, d_phase[cur], 1);

    #pragma omp parallel for
    for (int i = 0; i < num_subapertures; i++) {
        compute_wcom(&ctx, k + 1, i, h_slopes[prev]);
    }

    cudaStreamSynchronize(stream_recon);
    dispatch_to_dm(d_phase[cur]);
}
```

Worst-case latency is bounded by profiling each stage on target hardware (CUDA events) and verifying the 99.9th-percentile frame time against the 10 ms budget — not just the mean, since AO control stability depends on bounded *worst-case* jitter. The Auto-PINN champion/challenger swap (Section 5.3.5) is included in this profiling regime: any candidate network whose p99.9 inference time would erode the margin is rejected at promotion time, never at runtime.

---

### 7. Highly Optimized CPU Control Engine (C Implementation)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>

#define ALIGN64 __attribute__((aligned(64)))

typedef struct {
    int num_subapertures;
    int num_actuators;
    int total_frames;

    unsigned char **cached_bmp_frames;

    double *R_matrix      ALIGN64;
    double *H_inv_matrix  ALIGN64;
    double *kalman_A      ALIGN64;
    double *kalman_Kinf   ALIGN64;

    double *slopes        ALIGN64;
    double *phase_map     ALIGN64;
    double *modal_state   ALIGN64;
    double *dm_strokes_lin   ALIGN64;
    double *dm_strokes_final ALIGN64;
    double *pinn_uncertainty ALIGN64;   // per-actuator epistemic sigma, from Auto-PINN ensemble

    uint8_t *bad_subap_mask;
    double   stroke_limit;
    double   rate_limit;
    double   prev_command[4096];

    void   *active_inference_engine;     // atomically swapped pointer (champion network)
} AOControlPipeline;

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

/* Combines command magnitude with Auto-PINN epistemic uncertainty (5.3.4/5.4):
 * a large, low-confidence correction is penalized more than a large,
 * high-confidence one. */
static inline double uncertainty_weighted_anomaly_score(AOControlPipeline *ctx,
                                                          double *u) {
    double score = autoencoder_anomaly_score(u);
    double mean_unc = 0.0;
    for (int i = 0; i < ctx->num_actuators; i++) mean_unc += ctx->pinn_uncertainty[i];
    mean_unc /= ctx->num_actuators;
    return score * (1.0 + GAMMA_UNCERTAINTY * mean_unc);
}

void ao_pipeline_execute_frame(AOControlPipeline *ctx, int frame_index) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ctx->num_subapertures; i++) {
        if (ctx->bad_subap_mask[i]) continue;
        // ... WCoM with adaptive window + correlation fallback inserted here ...
    }

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->num_actuators, 2 * ctx->num_subapertures,
                1.0, ctx->R_matrix, 2 * ctx->num_subapertures,
                ctx->slopes, 1,
                0.0, ctx->phase_map, 1);

    kalman_predict_update(ctx->kalman_A, ctx->kalman_Kinf,
                           ctx->modal_state, ctx->phase_map);

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->num_actuators, ctx->num_actuators,
                -1.0, ctx->H_inv_matrix, ctx->num_actuators,
                ctx->phase_map, 1,
                0.0, ctx->dm_strokes_lin, 1);

    // Auto-PINN inference: ensemble forward pass through the currently
    // promoted "champion" engine (atomic pointer, hot-swapped by the
    // background orchestrator in Sec 5.3.5; never blocks this thread).
    get_pinn_residual_with_uncertainty(ctx->active_inference_engine,
                                        ctx->slopes,
                                        ctx->dm_strokes_final,   // residual mean
                                        ctx->pinn_uncertainty);  // residual epistemic sigma
    #pragma omp simd
    for (int i = 0; i < ctx->num_actuators; i++)
        ctx->dm_strokes_final[i] += ctx->dm_strokes_lin[i];

    if (uncertainty_weighted_anomaly_score(ctx, ctx->dm_strokes_final) > ANOMALY_THRESHOLD) {
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

- **Interaction-matrix calibration:** automated poke-matrix routine (push-pull per actuator) with outlier-robust fitting, run offline and on-demand; feeds $H^{-1}$, the bad-actuator mask, and the Auto-PINN meta-learning task distribution (Section 5.3.3).
- **Hardware-in-the-loop benchmarking:** worst-case (p99.9) and mean per-stage latency logged via `clock_gettime(CLOCK_MONOTONIC)` / CUDA events, regression-tested against the 10 ms budget on every build *and* on every Auto-PINN champion promotion.
- **Unit/integration tests:** synthetic Kolmogorov phase screens (FFT or Zernike-sum method) validate Strehl recovery and reconstruction-matrix correctness against known ground truth; synthetic drift scenarios (simulated thermal ramps, actuator aging curves) validate the Auto-PINN drift-detection and meta-adaptation path before it is ever exposed to on-sky data.
- **Telemetry stream:** $r_0$, $\tau_0$, Cn²(h) bins, Kalman innovation residuals, anomaly scores, per-actuator clamp-activation counts, reconstruction condition number, Auto-PINN ensemble uncertainty, drift-score history, and champion/challenger promotion log are all retained at reduced rate for offline diagnosis and future model retraining.

---

### 9. Summary of Additions Over Baseline

1. Adaptive WCoM window sizing (Cramér–Rao-informed) + correlation-based low-SNR fallback.
2. Bad/saturated subaperture masking with low-rate reconstructor updates.
3. Truncated-SVD regularization with condition-number health monitoring, shared into Auto-PINN drift detection.
4. Matrix-free PCG path for ELT-scale actuator counts.
5. Kalman/LQG predictive control layer with online vibration-peak rejection.
6. Cn²(h) altitude-resolved turbulence profiling (SLODAR-style).
7. Transformer-based short-horizon turbulence forecasting feeding the Kalman model.
8. Hysteresis-aware, PDE-residual PINN loss for piezo-stack non-linearity.
9. **Auto-PINN: latency-constrained NAS (DARTS-style) for the residual corrector's architecture.**
10. **Auto-PINN: PBT/Bayesian hyperparameter and loss-weight optimization against live telemetry.**
11. **Auto-PINN: MAML/Reptile meta-initialization for sub-minute few-shot recalibration on detected drift.**
12. **Auto-PINN: ensemble-based epistemic uncertainty, fed into the hard safety layer's anomaly veto.**
13. **Champion/challenger shadow-mode deployment with statistically gated, latency-verified promotion.**
14. Independent hard safety layer: stroke/rate limiting + uncertainty-weighted autoencoder anomaly veto.
15. GPU/cuBLAS double-buffered streaming variant with worst-case latency profiling, extended to cover network hot-swaps.
16. Calibration, HIL benchmarking, and synthetic-phase-screen / synthetic-drift test framework.
