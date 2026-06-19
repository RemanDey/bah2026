# PROBLEM STATEMENT 9

## Developing and Optimizing Algorithms for Wavefront Reconstruction and Turbulence Characterization Using Shack-Hartmann Wavefront Sensor (SH-WFS) Time-Series Data
![Alternative text](dxh.webp)
### Description
Turbulence in the atmosphere distorts a plane-parallel wavefront propagating through it. An SH-WFS samples this distorted wavefront using an array of small lenslets. This Microlens Array (MLA) creates a spot-field on the detector and the deviation of these spots from their reference position is used to derive the reconstructed wavefront and its associated zernike coefficients. The conjugate of this reconstructed wavefront is typically used to derive an actuator map in units of the actuator stroke length which is then fed to a deformable mirror (DM) to correct for this distortion in real-time.

### Objectives
Objective is to use SH-WFS frames collected during turbulence simulated in the laboratory and to develop image processing algorithms to perform wavefront reconstruction, turbulence characterization and actuator map determination for the deformable mirror. Atmospheric turbulence has an inherent coherence timescale of the order of milliseconds and so, the reconstruction algorithm must be fast enough to measure the distortion and correct for it. Statistical parameters characterizing the strength of turbulence, like the Fried parameter ($r_0$) and the coherence time ($\tau_0$), are also to be derived from the same data.

### Expected Outcomes
- Reconstructed wavefront phase maps ($W(x_i, y_i)$) for each SH-WFS frame and the characterized turbulence in terms of the Fried parameter ($r_0$) and the coherence time ($\tau_0$).
- Actuator maps ($A(x_i, y_i)$ for each reconstructed wavefront map) in terms of the actuator stroke length of the deformable mirror.
- Incorporation of the effect of inter-actuator coupling while deriving these actuator maps.

### Data Required
- A time series of SH-WFS frames (essentially `.bmp` files) captured at a time interval of a few milliseconds by a science-grade camera.
- Basic information of frame data: pixel size and frame resolution.
- Basic information of MLA: size, number of lenslets and focal length.
- Pupil size of the turbulated beam.
- DM information and interactuator coupling. This dataset shall be provided.

### Suggested Tools/Technologies
Since the corrections are to be applied at a rate faster than at which the atmosphere changes (~10ms), it is advised to develop and optimize algorithms for a low-level language like C. Several methods exist in literature to perform wavefront reconstruction like zonal/modal methods using orthogonal polynomials or direct integration. Several open-source libraries also exist which maybe utilized to perform/optimize complex mathematical computations.

### Expected Solution / Steps to be followed to achieve the objectives
1. For each WFS frame, identify the centroid position of each spot associated with a sub-aperture using a suitable centroiding algorithm.
2. For each spot, calculate the spot deviation from its reference position.
3. Utilize an existing technique (e.g. zonal/modal methods) from the literature or develop your own algorithm to reconstruct the wavefront phase map. Student must consider that the actuator grid of DM and the lenslet grid of MLA are arranged in a Fried geometry.
4. Use the reconstructed maps or associated orthogonal basis coefficients (e.g. Zernike coefficients) to derive turbulence characteristics.
5. Use the conjugate of the reconstructed wavefront to derive an actuator map in terms of the actuator stroke lengths keeping in mind the inter-actuator coupling.

### Evaluation Criteria
- Reconstruction of the wavefront phase maps ($W(x_i, y_i)$) for each frame that conform to the turbulence characteristics.
- Deriving statistical parameters indicating the strength of turbulence.
- Speed and computational efficiency of the developed algorithms.
