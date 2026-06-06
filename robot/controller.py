import numpy as np
from robot.kinematics import (
    forward_kinematics,
    compute_jacobian,
    get_tool_points,
    joint_limit_gradient
)


class RCMController:
    def __init__(self, K1=5.0, K2=3.0, lam=0.01):
        # K1: gain on trocar error (primary task)
        # K2: gain on tip tracking (secondary task)
        # lam: damping factor for DLS pseudoinverse
        self.K1  = K1
        self.K2  = K2
        self.lam = lam

    def damped_pinv(self, J):
        # Damped least squares: Jt(JJt + lam^2 * I)^-1
        # Keeps joint velocities bounded near singularities
        return J.T @ np.linalg.inv(J @ J.T + self.lam**2 * np.eye(J.shape[0]))

    def null_space(self, J, J_pinv):
        # N = I - J†J
        # Any vector projected through N produces zero velocity at the trocar
        # (approximately — damped LS introduces small leakage)
        return np.eye(J.shape[1]) - J_pinv @ J

    def step(self, q, p_rcm_target, p_tip_target, dt):
        # One control step. Returns updated joint angles and both error vectors.
        q = np.asarray(q, dtype=float)

        # FK
        T, transforms = forward_kinematics(q)
        T_wrist = transforms[5]
        p_trocar, p_tip = get_tool_points(T_wrist)

        # Errors
        e_rcm = p_rcm_target - p_trocar
        e_tip = p_tip_target - p_tip

        # Jacobians for trocar and tip
        J_rcm = compute_jacobian(q, transforms, p_target=p_trocar)
        J_tip = compute_jacobian(q, transforms, p_target=p_tip)

        # Damped pseudoinverses
        J_rcm_pinv = self.damped_pinv(J_rcm)
        J_tip_pinv = self.damped_pinv(J_tip)

        # Null space of the RCM Jacobian
        N = self.null_space(J_rcm, J_rcm_pinv)

        # Primary task: drive trocar error to zero
        # Note: this enforces the trocar as a point-position constraint only.
        # A stricter formulation would also constrain the shaft line geometry,
        # but point-position is the standard approach for software-defined RCM.
        dq_primary = J_rcm_pinv @ (self.K1 * e_rcm)

        # Secondary task: tip tracking + joint limit avoidance, both in null space
        dq_tip    = J_tip_pinv @ (self.K2 * e_tip)
        dq_limits = joint_limit_gradient(q)
        dq_secondary = N @ (dq_tip + 0.1 * dq_limits)

        # Combine and integrate
        dq = dq_primary + dq_secondary
        q_new = q + dq * dt

        return q_new, e_rcm, e_tip


class NaiveController:
    # Baseline — plain pseudoinverse, tip tracking only, no null space, no damping.
    # Included to show what happens without the RCM constraint enforcement.
    def __init__(self, K=5.0):
        self.K = K

    def step(self, q, p_rcm_target, p_tip_target, dt):
        q = np.asarray(q, dtype=float)

        T, transforms = forward_kinematics(q)
        T_wrist = transforms[5]
        p_trocar, p_tip = get_tool_points(T_wrist)

        e_rcm = p_rcm_target - p_trocar
        e_tip = p_tip_target - p_tip

        # Tip Jacobian only, no RCM constraint
        J_tip = compute_jacobian(q, transforms, p_target=p_tip)

        # Plain pseudoinverse — no damping, will blow up near singularities
        J_tip_pinv = np.linalg.pinv(J_tip)

        dq = J_tip_pinv @ (self.K * e_tip)
        q_new = q + dq * dt

        return q_new, e_rcm, e_tip
