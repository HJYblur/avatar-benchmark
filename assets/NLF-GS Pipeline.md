### NLF-GS Pipeline 

##### Avatar template

The representation is based on SMPL-X model with $V$ vertices $E$ faces. We plan to attach $k$ Gaussians to each face, thus there are $E*k$ Gaussians in total on the avatar, we represent the set as $G$. Following the [Original 3D Gaussian paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), we use the following properties to define a 3D Gaussian:
$$
Gaussian(g) = \{index, translation, rotation, scale, opacity, SH\},\ g\in G
$$
Among them, the index is given based on the face they're attached to, translation $t_p \in R^3$ refers to the relative translation between the gaussian center and the face center, along with rotation and scale it determines the shape information. Meanwhile, scale and opacity determines the appearance information. 

The goal of the model is to predict the shape and appearance properties of each Gaussian for once. Then in inference time, by predicting the params of SMPL-X model as the pose information, can we rig the Gaussians attached to it to generate drivable avatar. 

##### Input

Given a RGB image $I \in R^{H*W*3}$, the image will be sent to feature encoder (e.g. EfficientNetV2, DINOv2, etc) to generate features $f_{I} \in R^{h*w*C}$.

##### Training Pipeline

For every gaussian $g$ in the avatar template Gaussian Set $G$, we want to estimate its properties using two paths, namely Neural Localiser Field  for pose information, and Gaussian Estimator for shape and appearance information. 

Following the same definition in NLF paper, the localizer field of functions $\Phi_{pose}$ associates a function $f_{pose}(g)$ of every gaussian to predict the pose information. If we define the weight prediction function as $\omega_{pose}$, we have:
$$
(W_{pose}, b_{pose}) := \omega_{pose}(g)\\
(H_{2D}, H_{3D}, U) := conv_{1*1}(f_{pose}; W_{pose}, b_{pose})\\
p'_{3D} := soft-argmax(H_{3D})\\
p'_{2D} := soft-argmax(H_{2D})\\
u := \sum_{x, y} U_{x, y} softmax(H_{2D})_{x,y}
$$
$p'_{3D}$ refers to the estimated 3D coordinates of the canonical space, $p'_{2D}$ refers to the estimated 2D coordinates of the gaussian on the feature map, while $H_{3D}$ and $H_{2D}$ are the heatmaps of them.

Thus, we get the encoded feature of the targeted gaussian by sampling the feature map using  $p'_{2D}$, let's represent it as:
$$
feature_g := Sample(f_{I}; p'_{2D})
$$
we can use another two specific MLPs to decode the shape and appearance from there:
$$
translation, rotation, scale := MLP_{shape}(feature_g)\\
opacity, SH := MLP_{appearance}(feature_g)
$$

##### Conclusion

Thus, we can obtain the properties of the whole Gaussian Set. In inference time, just applying NLF to estimate the SMPL-X param can we drive the avatar.



