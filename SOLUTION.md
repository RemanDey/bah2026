# SOLUTION.md

## High-Performance Wavefront Reconstruction and Turbulence Characterization Pipeline for Shack-Hartmann WFS Time-Series Data

This document provides a research-grade, low-latency framework designed to process Shack-Hartmann Wavefront Sensor (SH-WFS) frames, calculate wavefront gradients, perform modal/zonal phase reconstruction, characterize atmospheric parameters ($r_0$, $\tau_0$), and derive cross-talk decoupled actuator stroke profiles under **Fried geometry** at processing cycles bounded within $\le 10\text{ ms}$.

---

### 1. High-Speed Sub-Pixel Centroiding & Slope Estimation

To maximize noise immunity and maintain structural consistency within a low-level implementation, each frame must undergo adaptive windowing and background clipping before calculating pixel moments.

#### 1.1 Dynamic Thresholding
Let $I(x,y)$ be the raw pixel intensity within a sub-aperture region of interest (ROI). To eliminate the impact of stray background illumination and detector dark currents, apply a regional floor filter:

$$I_{\text{th}}(x,y) = \max\left(0, I(x,y) - I_{\text{bg}} - k \cdot \sigma_{\text{bg}}\right)$$

Where $I_{\text{bg}}$ represents the mean local background intensity, $\sigma_{\text{bg}}$ is the spatial RMS noise of the unilluminated detector, and $k \in [2, 3]$ is a customizable sigma-clipping multiplier.

#### 1.2 Center of Mass (CoM) Formulation
The sub-pixel spot centers $(x_c, y_c)$ for each lenslet sub-aperture are evaluated using intensity-weighted primary moments:

$$x_c = \frac{\sum_{x,y} x \cdot I_{\text{th}}(x,y)}{\sum_{x,y} I_{\text{th}}(x,y)}, \quad y_c = \frac{\sum_{x,y} y \cdot I_{\text{th}}(x,y)}{\sum_{x,y} I_{\text{th}}(x,y)}$$

#### 1.3 Local Wavefront Gradients
Using reference coordinates $(x_{\text{ref}}, y_{\text{ref}})$ measured from a flat, plane-parallel reference beam, spatial phase derivative averages $(s_x, s_y)$ are computed from focal-plane deviations:

$$s_x = \frac{\partial \phi}{\partial x} \approx \frac{x_c - x_{\text{ref}}}{f_{\text{MLA}}}, \quad s_y = \frac{\partial \phi}{\partial y} \approx \frac{y_c - y_{\text{ref}}}{f_{\text{MLA}}}$$

Where $f_{\text{MLA}}$ is the calibrated focal length of the Microlens Array. 

For a frame consisting of $N$ sub-apertures, the gradients are grouped into a static column vector:

$$\mathbf{s} = \begin{bmatrix} s_x^1, s_x^2, \dots, s_x^N, s_y^1, s_y^2, \dots, s_y^N \end{bmatrix}^T \in \mathbb{R}^{2N \times 1}$$

---

### 2. Dual Reconstructive Architectures

#### 2.1 Modal Wavefront Reconstruction (Zernike Polynomial Basis)
Modal expansion represents the continuous phase map $\phi(\rho, \theta)$ across a circular pupil of radius $R$ as a linear combination of orthogonal Zernike modes:

$$\phi(\rho, \theta) = \sum_{j=1}^{M} a_j Z_j(\rho, \theta)$$

An Interaction Matrix $G \in \mathbb{R}^{2N \times M}$ maps these coefficient modes to the measured slope vector $\mathbf{s}$:

$$G = \begin{bmatrix} \frac{\partial Z_1}{\partial x} & \dots & \frac{\partial Z_M}{\partial x} \\ \frac{\partial Z_1}{\partial y} & \dots & \frac{\partial Z_M}{\partial y} \end{bmatrix}_{2N \times M} \implies \mathbf{s} = G \mathbf{a}$$

Because $G$ is non-square and inherently filters out the piston mode ($Z_1$), a regularized control matrix $R_{\text{modal}}$ is built offline using the Moore-Penrose Pseudo-Inverse via Singular Value Decomposition (SVD):

$$R_{\text{modal}} = (G^T G)^{-1} G^T$$

The real-time coefficient vector estimation is completed in a single matrix-vector multiplication step:

$$\mathbf{a} = R_{\text{modal}} \cdot \mathbf{s}$$

#### 2.2 Zonal Wavefront Reconstruction (Fried Geometry Matrix)
Under **Fried Geometry**, the phase points $\phi_{i,j}$ (which correspond directly to the deformable mirror's actuator grid nodes) are arranged at the *four vertices* surrounding each square lenslet sub-aperture center. 

The discrete finite-difference mappings relate the phase values at the vertices to the average sub-aperture gradients:

$$s_x^{i,j} = \frac{\phi_{i+1, j} + \phi_{i+1, j+1} - \phi_{i, j} - \phi_{i, j+1}}{2 \cdot d}$$

$$s_y^{i,j} = \frac{\phi_{i, j+1} + \phi_{i+1, j+1} - \phi_{i, j} - \phi_{i+1, j}}{2 \cdot d}$$

Where $d$ represents the sub-aperture grid step width. Assembling this system into a sparse matrix equation yields $\mathbf{s} = D \boldsymbol{\phi}$. The phase values at each node are recovered using:

$$R_{\text{zonal}} = (D^T D)^{-1} D^T \implies \boldsymbol{\phi} = R_{\text{zonal}} \cdot \mathbf{s}$$

---

### 3. Statistical Turbulence Characterization

By maintaining a sliding temporal queue of estimated Zernike coefficient states, the statistical parameters of the laboratory-simulated turbulence can be calculated asynchronously.

#### 3.1 Fried Parameter ($r_0$) Derivation
Using Noll’s residual variance formulations derived from Kolmogorov turbulence theory, the temporal variance of the tip-tilt tracking terms ($a_2, a_3$) over a pupil diameter $D_{\text{pupil}}$ can be directly mapped to the structural scaling parameter:

$$\sigma_{\text{tilt}}^2 = \langle a_2^2 \rangle + \langle a_3^2 \rangle = 0.896 \left(\frac{D_{\text{pupil}}}{r_0}\right)^{5/3}$$

Inverting this expression across the time-series history yields:

$$r_0 = D_{\text{pupil}} \cdot \left( \frac{0.896}{\sigma_{\text{tilt}}^2} \right)^{3/5}$$

#### 3.2 Coherence Time ($\tau_0$) Derivation
The temporal autocovariance profile $C_a(\Delta t)$ of the dominant reconstructed modes is monitored over an incremental frame delay $\Delta t$:

$$C_a(\Delta t) = \langle a(t) \cdot a(t + \Delta t) \rangle$$

The characteristic coherence time $\tau_0$ corresponds to the exact delay offset where the normalized temporal autocorrelation drops below the standard $1/e$ threshold:

$$\frac{C_a(\tau_0)}{C_a(0)} = \frac{1}{e} \approx 0.3679$$

---

### 4. Actuator Map Inversion & Mechanical Decoupling

To correct for aberrations in real time, the deformable mirror profile must apply the exact conjugate shape of the reconstructed phase map: $\mathbf{A}_{\text{target}} = -\boldsymbol{\phi}$.

#### 4.1 Inter-Actuator Coupling Matrix ($H$)
Due to the mechanical stiffness of the mirror faceplate, driving an individual actuator influences its nearest neighbors. The actual surface displacement vector $\mathbf{A}$ relates to the applied stroke command vector $\mathbf{u}$ through a symmetric influence matrix $H$:

$$A_i = u_i + \alpha \sum_{j \in N(i)} u_j \implies \mathbf{A} = H \mathbf{u}$$

Where $\alpha$ represents the inter-actuator mechanical coupling factor (typically $\alpha \in [0.10, 0.20]$) and $N(i)$ lists the structural orthogonal neighbors of actuator $i$.

#### 4.2 Decoupled Real-Time Commands
To compensate for cross-talk and achieve precise phase conjugation, the raw influence system must be inverted:

$$\mathbf{u} = H^{-1} \mathbf{A}_{\text{target}} = H^{-1} \cdot (-\boldsymbol{\phi})$$

Since $H$ depends strictly on static hardware constraints, $H^{-1}$ can be pre-computed at startup. This reduces the real-time decoupling stage to a fast matrix-vector multiplication.

---

### 5. Highly Optimized Control Engine Architecture (C Implementation)

The core execution pipeline avoids dynamic memory allocations (`malloc`/`free`) inside the runtime loop and relies on highly optimized BLAS routines for rapid matrix operations.

```c
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

typedef struct {
    int num_subapertures;
    int num_actuators;
    double *R_matrix;        // Pre-computed Reconstruction Matrix [num_actuators x 2*num_subapertures]
    double *H_inv_matrix;    // Pre-computed Inverse Coupling Matrix [num_actuators x num_actuators]
    double *slopes;          // Runtime measured gradient vector [2 * num_subapertures]
    double *phase_map;       // Intermediate reconstructed phase [num_actuators]
    double *dm_strokes;      // Final decoupled actuator commands [num_actuators]
} AOControlPipeline;

/**
 * Executes real-time phase reconstruction and actuator command decoupling.
 * Bounded runtime complexity: O(N_actuators * N_slopes) + O(N_actuators^2)
 */
void ao_pipeline_execute_frame(AOControlPipeline *ctx) {
    // Step 1: Reconstruct Phase Map using BLAS GEMV: phase_map = 1.0 * R_matrix * slopes + 0.0
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                ctx->num_actuators, 2 * ctx->num_subapertures, 
                1.0, ctx->R_matrix, 2 * ctx->num_subapertures, 
                ctx->slopes, 1, 
                0.0, ctx->phase_map, 1);

    // Step 2: Decouple Cross-Talk & Apply Conjugate Mirror Commands: dm_strokes = -1.0 * H_inv_matrix * phase_map + 0.0
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                ctx->num_actuators, ctx->num_actuators, 
                -1.0, ctx->H_inv_matrix, ctx->num_actuators, 
                ctx->phase_map, 1, 
                0.0, ctx->dm_strokes, 1);
}
