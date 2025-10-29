## SMPL Param
| Source file                | Field                   | Used in `smpl.forward()`                                       | Meaning                                                                                                             |
| -------------------------- | ----------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `consensus.pkl`            | `betas`                 | `betas`                                                        | Shape coefficients **β** (10D). Define person’s static body shape.                                                  |
| `reconstructed_poses.hdf5` | `pose`                  | split into `global_orient` (first 3) and `body_pose` (next 69) | Pose parameters **θ**. 24 joints × 3 axis-angle rotation vectors.                                                   |
| `reconstructed_poses.hdf5` | `trans`                 | `transl`                                                       | Global translation **T** in world coordinates.                                                                      |
| `consensus.pkl`            | `v_personal` (optional) | added manually to mesh vertices                                | Per-vertex offsets for personalized geometry; not part of the standard SMPL equation but used for local correction. |



Connections to Paper:

Paper formula:
$$
W(T_P(\beta, \theta; \bar{T}, S, P), J(\beta; \mathcal{J}, \bar{T}, S), \theta, \mathcal{W})
$$

| Symbol       | Name                                                | Corresponding code/data                                                           |
| ------------ | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| **β**        | Shape coefficients                                  | `betas` from `consensus.pkl`                                                      |
| **θ**        | Pose coefficients                                   | `pose` from `reconstructed_poses.hdf5` (split into `global_orient` + `body_pose`) |
| **T̄**       | Template mesh in rest (T-pose)                      | internal to SMPL model (stored in `.pkl` model file)                              |
| **S**        | Shape blend shapes                                  | internal fixed bases in the SMPL model                                            |
| **P**        | Pose blend shapes                                   | internal bases (used for pose-dependent deformations)                             |
| **J(β)**     | Joint regressor output                              | computed by SMPL using the betas (joint locations depend on body shape)           |
| **θ̄, W(·)** | Linear blend skinning weights and blending function | fixed inside SMPL; applied automatically when you call `model()`                  |
| **transl**   | Global translation                                  | applied after deformation to move the mesh into the camera/world position         |


## Camera Param

Concise coordinate explanation.

### PyTorch3D coordinate system

Right-handed, camera-centric when you use `look_at_view_transform`.

* **+X:** points **right** in the image.
* **+Y:** points **up** in the image.
* **+Z:** points **away from the camera** (into the screen).
* Object space: when you build your mesh, it lives roughly around the origin.
  The camera orbits around that origin using the parameters below.

---

### `look_at_view_transform(dist, elev, azim)`

It places the **camera** on a virtual sphere of radius `dist` centered at the object.
Then it aims the camera at the origin.

* **dist** – distance from camera to object.
* **elev** – vertical angle (degrees) above/below the *x–z plane*.

  * `0` → camera level with the object.
  * `+90` → camera directly above, looking down.
  * `–90` → camera below, looking up.
* **azim** – rotation around the *z-axis*, measured in degrees.

  * `0` → camera on +Y axis, looking toward origin.
  * `90` → on +X axis.
  * `180` → on –Y axis (opposite side).
  * `–90` → on –X axis.



| Parameter  | Meaning                                      | Typical range                            |
| ---------- | -------------------------------------------- | ---------------------------------------- |
| `cam_dist` | Distance from camera to object center        | ~2–4                                     |
| `cam_elev` | Elevation angle (degrees above the XY plane) | 0 = horizontal, 30 = slightly above      |
| `cam_azim` | Azimuth angle (degrees around the object)    | 0 = camera in front, 180 = camera behind |
