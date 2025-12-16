## Architecture

### 1. Avatar Template & Gaussian Definition

The avatar is represented by a set of Gaussians \( \mathcal{G} \) attached to the faces of a canonical SMPL-X mesh.
Let the mesh faces be \( \mathcal{F}_{\text{mesh}} \). We attach \( k \) Gaussians per face, resulting in \( N = k \cdot |\mathcal{F}_{\text{mesh}}| \) Gaussians in total.

Each Gaussian \( g_i \in \mathcal{G} \) in canonical (T-pose) space is defined as:
\[
g_i =
\left\{
\mathbf{x}_i^{\text{can}},
\mathbf{r}_i,
\mathbf{s}_i,
\alpha_i,
\mathbf{c}_{i}^{SH}
\right\}
\]

where:
- \(\mathbf{x}_i^{\text{can}} \in \mathbb{R}^3\): canonical Gaussian center, defined via barycentric coordinates on the corresponding mesh face,
- \( \mathbf{r}_i \in \mathbb{R}^4 \): rotation quaternion,
- \( \mathbf{s}_i \in \mathbb{R}^3 \): anisotropic scale,
- \( \alpha_i \in \mathbb{R} \): opacity,
- \( \mathbf{c}_{i}^{SH} \): spherical harmonics coefficients (initially DC only).

The template defines a fixed canonical topology; all pose and appearance variations are introduced through conditioning and decoding.

---

### 2. Image Encoding and Feature Extraction

Given an input RGB image
\[
I \in \mathbb{R}^{H \times W \times 3},
\]
a shared backbone encoder extracts dense image features:
\[
F_{\text{img}} = \text{Backbone}(I),
\quad
F_{\text{img}} \in \mathbb{R}^{C \times H_f \times W_f}.
\]

In parallel, a Neural Localizer Field (NLF) predicts SMPL-X related quantities:
\[
\mathcal{P} = \text{NLF}(I),
\]
where \( \mathcal{P} \) includes SMPL-X pose parameters as well as estimated vertex positions in both 2D and 3D space.

---

### 3. Identity Encoding

To enable cross-subject generalization, we extract a **global identity latent** from the image features.
We pool backbone features over the estimated avatar foreground:
\[
\mathbf{z}_{id}
=
\phi_{id}
\left(
\frac{1}{|\Omega|}
\sum_{(u,v)\in \Omega} F_{\text{img}}(u,v)
\right),
\quad
\mathbf{z}_{id} \in \mathbb{R}^{D}.
\]

Here, \( \Omega \) denotes the set of feature-map 2d locations corresponding to the avatar foreground, obtained via SMPL-X projections.
The identity code \( \mathbf{z}_{id} \) captures view-invariant appearance attributes (e.g., clothing, hair, and overall body style) and is shared across all Gaussians of the same avatar.

In practice, the pooling over \( \Omega \) is approximated by sampling a fixed number \( N_{\text{identity}} \) (e.g., 1024) of projected foreground points.

---

### 4. Pose-Conditioned Gaussian Localization

Each Gaussian \(g_i\) is associated with a parent mesh face whose vertices have 3D positions $\{\mathbf{v}_{i1}^{3D}, \mathbf{v}_{i2}^{3D}, \mathbf{v}_{i3}^{3D}\}$. 

Let
$$
 \mathcal{B}_i = (b_{i1}, b_{i2}, b_{i3}) 
$$
denote the barycentric weights of \(g_i\) on that face.

The posed 3D center of the Gaussian is computed as:
\[
\mathbf{x}_i^{3D}
=
\sum_{j=1}^{3} b_{ij}\,\mathbf{v}_{ij}^{3D},
\quad
\mathbf{x}_i^{3D} \in \mathbb{R}^{3}.
\]

Using the same barycentric weights, the corresponding 2D projection is obtained from the projected SMPL-X vertices:
\[
\mathbf{x}_i^{2D}
=
\sum_{j=1}^{3} b_{ij}\,\mathbf{v}_{ij}^{2D}.
\]

---

### 5. Local Feature Sampling

Local appearance features are sampled from the image feature map at the projected Gaussian locations:
\[
\mathbf{f}_i
=
\text{Sample}(F_{\text{img}}, \mathbf{x}_i^{2D}),
\quad
\mathbf{f}_i \in \mathbb{R}^{C}.
\]

This operation (implemented via bilinear sampling) captures view-dependent, high-frequency appearance cues for each Gaussian.

---

### 6. Identity-Conditioned Gaussian Decoding

The global identity latent is broadcast to all Gaussians and concatenated with local and pose features:
\[
\tilde{\mathbf{z}}_i
=
\left[
\mathbf{z}_{id},
\mathbf{f}_i,
\mathbf{x}_i^{3D}
\right]
\in \mathbb{R}^{D + C + 3}.
\]

A shared lightweight decoder predicts the Gaussian parameters:
\[
(\mathbf{r}_i, \mathbf{s}_i, \alpha_i, \mathbf{c}_{i}^{SH})
=
\phi_{\text{dec}}(\tilde{\mathbf{z}}_i).
\]

---

### 7. Rendering and Supervision

The posed Gaussians are rendered using a differentiable 3D Gaussian renderer:
\[
\hat{I} = \text{Render}\left(\{ g_i \}_{i=1}^{N}\right).
\]

Supervision plan is to train minimizes image-space reconstruction losses between the rendered image \(\hat{I}\) and the input image \(I\), augmented with perceptual and regularization terms.