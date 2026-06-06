# Mathematical Foundations of Remote Center of Motion (RCM) Control

This document presents the detailed mathematical derivations for the Forward Kinematics (FK), Jacobian, and constrained control formulation used in the RCM control package.

---

## 1. Robot Kinematics

Let the robot be represented as an $n$-degree-of-freedom (DoF) serial manipulator. The joint positions are denoted by the vector:
$$\mathbf{q} = \begin{bmatrix} q_1 & q_2 & \dots & q_n \end{bmatrix}^T \in \mathbb{R}^n$$

### Forward Kinematics (FK)
The forward kinematics maps the joint configuration $\mathbf{q}$ to the pose of the links. In surgical robotics, the end-effector is a long, slender tool shaft. Let:
- $\mathbf{p}_{\text{base}}(\mathbf{q}) \in \mathbb{R}^3$ be the position of the tool's base (where the instrument insertion mechanism starts).
- $\mathbf{p}_{\text{tip}}(\mathbf{q}) \in \mathbb{R}^3$ be the position of the instrument tip.
- $\mathbf{u}(\mathbf{q}) \in \mathbb{R}^3$ be the unit direction vector of the tool shaft pointing from the base to the tip:
  $$\mathbf{u}(\mathbf{q}) = \frac{\mathbf{p}_{\text{tip}}(\mathbf{q}) - \mathbf{p}_{\text{base}}(\mathbf{q})}{\|\mathbf{p}_{\text{tip}}(\mathbf{q}) - \mathbf{p}_{\text{base}}(\mathbf{q})\|}$$

Any point $\mathbf{p}(s, \mathbf{q})$ along the tool shaft can be parameterized by the distance $s \in [0, L]$ from the tool base:
$$\mathbf{p}(s, \mathbf{q}) = \mathbf{p}_{\text{base}}(\mathbf{q}) + s \mathbf{u}(\mathbf{q})$$
where $L(\mathbf{q}) = \|\mathbf{p}_{\text{tip}}(\mathbf{q}) - \mathbf{p}_{\text{base}}(\mathbf{q})\|$ is the current length of the instrument.

---

## 2. Remote Center of Motion (RCM) Constraint

The RCM constraint requires that the tool shaft passes through a fixed spatial point $\mathbf{p}_{\text{rcm}} \in \mathbb{R}^3$ (typically the incision site or trocar) at all times.

This means there exists some scalar $s_{\text{rcm}} \in [0, L]$ such that:
$$\mathbf{p}(s_{\text{rcm}}, \mathbf{q}) = \mathbf{p}_{\text{rcm}}$$

Equivalently, the perpendicular distance from the trocar point $\mathbf{p}_{\text{rcm}}$ to the tool shaft line must be zero:
$$\mathbf{e}_{\text{rcm}}(\mathbf{q}) = \left( \mathbf{I} - \mathbf{u}(\mathbf{q})\mathbf{u}(\mathbf{q})^T \right) \left( \mathbf{p}_{\text{rcm}} - \mathbf{p}_{\text{base}}(\mathbf{q}) \right) = \mathbf{0}$$

Where $\mathbf{e}_{\text{rcm}}(\mathbf{q}) \in \mathbb{R}^3$ represents the RCM tracking error (which is perpendicular to the tool shaft).

---

## 3. Differential Kinematics & Jacobians

To control the robot, we linearize the kinematics by computing the relationship between joint velocities $\mathbf{\dot{q}}$ and Cartesian velocities.

### Tool Base and Tip Jacobians
Let $\mathbf{J}_{\text{base}}(\mathbf{q}) \in \mathbb{R}^{3 \times n}$ and $\mathbf{J}_{\text{tip}}(\mathbf{q}) \in \mathbb{R}^{3 \times n}$ be the translational Jacobians of the tool base and tip respectively:
$$\mathbf{\dot{p}}_{\text{base}} = \mathbf{J}_{\text{base}}(\mathbf{q}) \mathbf{\dot{q}}$$
$$\mathbf{\dot{p}}_{\text{tip}} = \mathbf{J}_{\text{tip}}(\mathbf{q}) \mathbf{\dot{q}}$$

### Shaft Direction Jacobian
Differentiating the unit direction $\mathbf{u}(\mathbf{q})$ yields:
$$\mathbf{\dot{u}} = \mathbf{J}_u(\mathbf{q}) \mathbf{\dot{q}}$$
where $\mathbf{J}_u(\mathbf{q}) \in \mathbb{R}^{3 \times n}$ can be derived using the quotient rule on the vector:
$$\mathbf{J}_u(\mathbf{q}) = \frac{1}{\|\mathbf{d}\|} \left( \mathbf{I} - \mathbf{u}\mathbf{u}^T \right) \left( \mathbf{J}_{\text{tip}}(\mathbf{q}) - \mathbf{J}_{\text{base}}(\mathbf{q}) \right)$$
where $\mathbf{d} = \mathbf{p}_{\text{tip}} - \mathbf{p}_{\text{base}}$.

### RCM Point Jacobian
Differentiating the RCM position equation $\mathbf{p}_{\text{rcm}} = \mathbf{p}(s_{\text{rcm}}, \mathbf{q})$:
$$\mathbf{\dot{p}}_{\text{rcm}} = \mathbf{\dot{p}}_{\text{base}} + \dot{s}_{\text{rcm}} \mathbf{u} + s_{\text{rcm}} \mathbf{\dot{u}}$$
Substituting the Jacobians:
$$\mathbf{\dot{p}}_{\text{rcm}} = \left( \mathbf{J}_{\text{base}}(\mathbf{q}) + s_{\text{rcm}} \mathbf{J}_u(\mathbf{q}) \right) \mathbf{\dot{q}} + \mathbf{u} \dot{s}_{\text{rcm}}$$

We define the RCM Jacobian associated with the robot joints as:
$$\mathbf{J}_{\text{rcm\_joint}}(\mathbf{q}, s_{\text{rcm}}) = \mathbf{J}_{\text{base}}(\mathbf{q}) + s_{\text{rcm}} \mathbf{J}_u(\mathbf{q})$$

Thus:
$$\mathbf{\dot{p}}_{\text{rcm}} = \mathbf{J}_{\text{rcm\_joint}}(\mathbf{q}, s_{\text{rcm}}) \mathbf{\dot{q}} + \mathbf{u} \dot{s}_{\text{rcm}}$$

Since the RCM point is fixed, we require $\mathbf{\dot{p}}_{\text{rcm}} = \mathbf{0}$. We can isolate the RCM constraint by projecting perpendicular to the tool shaft (multiplying by $\mathbf{P}_{\perp} = \mathbf{I} - \mathbf{u}\mathbf{u}^T$), which eliminates the sliding velocity $\dot{s}_{\text{rcm}}$:
$$\left( \mathbf{I} - \mathbf{u}\mathbf{u}^T \right) \mathbf{J}_{\text{rcm\_joint}}(\mathbf{q}, s_{\text{rcm}}) \mathbf{\dot{q}} = \mathbf{0}$$

Let the **constrained RCM Jacobian** be:
$$\mathbf{J}_{\text{rcm\_const}}(\mathbf{q}, s_{\text{rcm}}) = \left( \mathbf{I} - \mathbf{u}\mathbf{u}^T \right) \mathbf{J}_{\text{rcm\_joint}}(\mathbf{q}, s_{\text{rcm}})$$

---

## 4. Control Formulation

We frame the RCM control problem as a **multi-task priority control** problem using either **Null-Space Projection** or **Weighted Least-Squares (WLS)**.

### Task 1: RCM Constraint (High Priority)
We want to drive the RCM error $\mathbf{e}_{\text{rcm}} \to \mathbf{0}$. We specify the desired RCM velocity as:
$$\mathbf{v}_{\text{rcm}} = -K_{\text{rcm}} \mathbf{e}_{\text{rcm}}$$
where $K_{\text{rcm}} > 0$ is a proportional gain. The joint velocity to satisfy this is solved via:
$$\mathbf{J}_{\text{rcm\_const}} \mathbf{\dot{q}} = \mathbf{v}_{\text{rcm}}$$

### Task 2: Tip Tracking (Lower Priority)
We want the tool tip to track a target trajectory $\mathbf{p}_{\text{tip, d}}(t)$ with velocity $\mathbf{v}_{\text{tip}} = \mathbf{\dot{p}}_{\text{tip, d}} - K_{\text{tip}} (\mathbf{p}_{\text{tip}} - \mathbf{p}_{\text{tip, d}})$.
$$\mathbf{J}_{\text{tip}} \mathbf{\dot{q}} = \mathbf{v}_{\text{tip}}$$

### Controller 1: Null-Space Projection (Strict Hierarchy)
To ensure the RCM constraint is *never* violated for the sake of tracking, we project the tip tracking task into the null-space of the RCM task:
$$\mathbf{\dot{q}} = \mathbf{J}_{\text{rcm\_const}}^{\dagger} \mathbf{v}_{\text{rcm}} + \left( \mathbf{I} - \mathbf{J}_{\text{rcm\_const}}^{\dagger} \mathbf{J}_{\text{rcm\_const}} \right) \mathbf{J}_{\text{tip}}^{\dagger} \mathbf{v}_{\text{tip}}$$
where $(\cdot)^{\dagger}$ represents the Moore-Penrose pseudoinverse.

### Controller 2: Weighted Least-Squares (Optimization-Based)
Alternatively, we can solve a quadratic program (QP) or a weighted least-squares problem:
$$\min_{\mathbf{\dot{q}}} \left( w_{\text{rcm}} \|\mathbf{J}_{\text{rcm\_const}} \mathbf{\dot{q}} - \mathbf{v}_{\text{rcm}}\|^2 + w_{\text{tip}} \|\mathbf{J}_{\text{tip}} \mathbf{\dot{q}} - \mathbf{v}_{\text{tip}}\|^2 + w_{\text{damp}} \|\mathbf{\dot{q}}\|^2 \right)$$
where:
- $w_{\text{rcm}}$ is a large weight (e.g., $10^4$) to enforce RCM strictness.
- $w_{\text{tip}}$ is a moderate weight (e.g., $1.0$) for tracking.
- $w_{\text{damp}}$ is a small regularization weight (e.g., $10^{-4}$) to prevent singularity issues.

The analytical solution to this optimization problem is:
$$\mathbf{\dot{q}} = \left( w_{\text{rcm}} \mathbf{J}_{\text{rcm\_const}}^T \mathbf{J}_{\text{rcm\_const}} + w_{\text{tip}} \mathbf{J}_{\text{tip}}^T \mathbf{J}_{\text{tip}} + w_{\text{damp}} \mathbf{I} \right)^{-1} \left( w_{\text{rcm}} \mathbf{J}_{\text{rcm\_const}}^T \mathbf{v}_{\text{rcm}} + w_{\text{tip}} \mathbf{J}_{\text{tip}}^T \mathbf{v}_{\text{tip}} \right)$$

This WLS form is highly robust, avoids algorithmic singularities, and can easily incorporate joint limit avoidance.
