# RCM Controller — Math Notes

These are the derivations behind `robot/kinematics.py` and `robot/controller.py`.

---

## 1. UR5 DH Parameters

The UR5 is modelled using Denavit-Hartenberg convention. Four parameters per joint:

| Parameter | Meaning |
|-----------|---------|
| a | Link length (along X) |
| d | Link offset (along Z) |
| α | Link twist (rotation around X) |
| θ | Joint angle (rotation around Z, this is the variable) |

Transform for one joint:
```
T = | cos θ   -sin θ·cos α    sin θ·sin α   a·cos θ |
    | sin θ    cos θ·cos α   -cos θ·sin α   a·sin θ |
    |   0         sin α          cos α          d    |
    |   0           0              0             1    |
```

UR5 values from the Universal Robots datasheet:

| Joint | a (m)    | d (m)    | α (rad) |
|-------|----------|----------|---------|
| 1     | 0.0      | 0.08916  | π/2     |
| 2     | -0.42500 | 0.0      | 0       |
| 3     | -0.39225 | 0.0      | 0       |
| 4     | 0.0      | 0.10915  | π/2     |
| 5     | 0.0      | 0.09465  | -π/2    |
| 6     | 0.0      | 0.0823   | 0       |

---

## 2. Forward Kinematics

Chain the six joint transforms:
```
T_0_to_6 = T1 × T2 × T3 × T4 × T5 × T6
```

End effector position is the top-right 3×1 column of the result.

Tool points come from the wrist frame T6. The tool shaft runs along the wrist
Z axis, so:
```
p_trocar = p_wrist + trocar_depth × u_shaft
p_tip    = p_wrist + tool_length  × u_shaft
```

u_shaft is the third column of T6's rotation block.

---

## 3. Geometric Jacobian

For revolute joint i, contribution to velocity of point p:
```
J_column_i = z_{i-1} × (p - p_{i-1})
```

z_{i-1} is the joint rotation axis (Z of frame i-1), p_{i-1} is its origin,
× is cross product. This comes from basic rotation kinematics — velocity of a
point due to rotation about an axis is the cross product of the axis direction
and the lever arm.

Full 3×6 Jacobian:
```
J = [J_col_1 | J_col_2 | J_col_3 | J_col_4 | J_col_5 | J_col_6]
```

Two Jacobians per timestep: J_rcm (trocar point) and J_tip (tool tip).

Note: position-only, so 3×6 not 6×6. Orientation control would need the full
6×6 geometric Jacobian with angular velocity rows.

---

## 4. Damped Least Squares

Plain pseudoinverse J† = Jᵀ(JJᵀ)⁻¹ breaks near singularities where det(JJᵀ)
approaches zero — the inversion produces huge joint velocities.

Fix: add a regularization term:
```
J†_damped = Jᵀ(JJᵀ + λ²I)⁻¹
```

The smallest eigenvalue of (JJᵀ + λ²I) is at least λ², so the inversion is
always stable. In SVD terms (J = UΣVᵀ):
```
σi → σi / (σi² + λ²)
```

When σi is large this is roughly 1/σi (same as before). When σi → 0 the
output goes to 0 rather than blowing up.

λ=0.01 was picked by sweeping [0.001, 0.01, 0.05, 0.1, 0.5]. Higher values
hurt accuracy, lower values risk instability.

---

## 5. Null-Space Controller

6 joints, 3 constraints (X/Y/Z of trocar point). That leaves 3 degrees of
freedom that don't affect the trocar — the null space of J_rcm.

Null space projector:
```
N = I - J†_rcm × J_rcm
```

For any v, J_rcm × (N × v) = 0. Proof for the idealized case (exact
Moore-Penrose pseudoinverse):
```
J_rcm × N × v
= J_rcm × (I - J†_rcm × J_rcm) × v
= (J_rcm - J_rcm × J†_rcm × J_rcm) × v
= (J_rcm - J_rcm) × v    ← because J×J†×J = J for exact pseudoinverse
= 0
```

Controller output:
```
dq = J†_rcm × (K1 × e_rcm)        ← primary: drive trocar error to zero
   + N × J†_tip × (K2 × e_tip)    ← secondary: tip tracking in null space
```

The secondary term gets projected through N so it can't affect the trocar
in the idealized case. With damped LS the projection isn't exact — there's
small leakage, which is why drift correction exists.

Note: the trocar constraint here is point-position only. A stricter
formulation would also constrain the shaft line to pass through the trocar
geometrically. This simplification is common in software-defined RCM work.

---

## 6. Singularity Detection

Manipulability measure:
```
w = sqrt(det(J × Jᵀ)) = σ1 × σ2 × σ3
```

σ1, σ2, σ3 are singular values from SVD. As w → 0 the robot approaches a
singularity. Damped LS keeps the controller from failing at these points.

---

## 7. Drift Correction

Euler integration accumulates error. Every `correction_interval` steps,
measure trocar drift directly:
```
e_drift = p_rcm_actual - p_rcm_target
```

If the norm exceeds `drift_threshold`, apply a correction:
```
dq_correction = J†_rcm × (K1 × e_drift)
q = q + dq_correction
```

This is a direct position fix, separate from the velocity loop. Fired 2 times
across 500 steps in the test run.

Note: integration uses first-order Euler (q ← q + dq·dt). Runge-Kutta would
reduce accumulated error at larger timesteps.

---

## 8. What's not modelled

This is kinematics only. No force modelling. Literature cites things like 31 N
lateral load tolerance and 5.6 Nm torque at the trocar as representative
benchmarks — a force-aware controller would need to account for these.

---

## 9. Results

Numbers from actual runs, not made up.

| Metric | Value |
|--------|-------|
| RCM error mean | 0.0777 mm |
| RCM error max | 0.5475 mm |
| RCM error std | 0.0717 mm |
| Tolerance from literature | 29.8 mm |
| Margin | 384x |
| Naive controller mean | 195.4 mm |
| Drift corrections (500 steps) | 2 |
| λ | 0.01 |
| dt | 0.01 s |
