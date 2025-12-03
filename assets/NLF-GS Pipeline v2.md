#### 1. Avatar Template & Definition

The avatar is represented by a set of Gaussians $G$ attached to the SMPL-X mesh faces. Let the canonical mesh have faces $F_{mesh}$. We attach $k$ Gaussians to each face.

Each Gaussian $g \in G$ is defined in the canonical space (T-pose) by:
$$
g = \{ \mathbf{x}_{can}, \mathbf{r}, \mathbf{s}, \alpha, \mathbf{c}_{SH}, \mathbf{i} \}
$$
Where

- Position ($\mathbf{x}_{can} \in \mathbb{R}^3$): The canonical center of the Gaussian. It is derived from the corresponding face center plus a learnable offset $\Delta x_p$ (initialized to 0).
- Rotation ($\mathbf{r} \in \mathbb{R}^4$): The rotation of the Gaussian, typically represented as a unit quaternion $(w, x, y, z)$ to ensure valid 3D rotations.
- Scale ($\mathbf{s} \in \mathbb{R}^3$): The anisotropic scaling factor, representing the extent of the Gaussian along its local axes.
- Opacity ($\alpha \in \mathbb{R}^1$): A scalar value representing the density of the Gaussian.
- Spherical Harmonics ($\mathbf{c}_{SH} \in \mathbb{R}^{3(d_{sh}+1)^2}$): The coefficients for view-dependent color. We propose to start with the DC component, which means $d_{sh}=0$ (gaussian only has diffuse color), then gradually transition to $d_{sh} = 3$ for better view-dependent visualization.
- Index ($i\in\mathbb{R}^1$): The index of the gaussian itself. 



#### 2. Input & Feature Extraction

Given an RGB image $I \in \mathbb{R}^{H \times W \times 3}$:
$$
F_{img} = Encoder(I), \quad F_{img} \in \mathbb{R}^{h \times w \times C}
$$


#### 3. Neural Localizer Field (Pose & Correspondence)



We treat the canonical coordinate $\mathbf{x}_{can}$ of a Gaussian $g$ as a query point to the NLF. The NLF acts as a Hypernetwork that generates a dynamic filter to find this point in the observation (image) space.

Let $\phi_{hyper}$ be the MLP that predicts filter weights:
$$
(W_g, b_g) := \phi_{hyper}(\mathbf{x}_{can})
$$
We apply these weights to the image feature map $F_{img}$ using a $1 \times 1$ convolution:
$$
(H_{2D}, H_{3D}, U) := Conv_{1 \times 1}(F_{img}; W_g, b_g)
$$

- $H_{2D} \in \mathbb{R}^{h \times w}$: 2D Heatmap of the Gaussian's location.
- $H_{3D}$: volumetric heatmap.
- $U$: Uncertainty map (optional, but part of NLF).

We retrieve the estimated 2D coordinates $\mathbf{p}'_{2D}$ of the feature map and the corresponding 3D position $p'_{3D}$ via soft-argmax:
$$
\mathbf{p}'_{2D} = \text{SoftArgmax}(H_{2D})\\
\mathbf{p}'_{3D} = \text{SoftArgmax}(H_{3D})
$$


#### 4. Gaussian Estimator (Shape & Appearance)



We sample the image features at the predicted 2D location to obtain a local descriptor for the Gaussian:
$$
f_g := \text{Sample}(F_{img}, \mathbf{p}'_{2D})
$$
We then decode the Gaussian attributes. We start with fixating the Gaussian center on the face center, thus the $\Delta \mathbf{x}_p$ is set to be zero. However, this may decrease the fidelity of the avatar, thus we could consider adding the prediction of the offset in the pipeline later. Note that we predict the offsets, not the absolute position (which is handled by the SMPL-X).
$$
\mathbf{r}, \mathbf{s} := MLP_{shape}(f_g)\\
\alpha, \mathbf{c}_{SH} := MLP_{appearance}(f_g)
$$


