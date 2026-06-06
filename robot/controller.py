import numpy as np
from robot.kinematics import (
    forward_kinematics,
    compute_jacobian,
    get_tool_points
)

class RCMController:
    def __init__(self, K1=5.0, K2=3.0, lam=0.01):
        """
        Parameters:
            K1  : gain for RCM constraint (primary task)
            K2  : gain for tip tracking (secondary task)
            lam : damping factor for damped least squares
        """
        self.K1  = K1
        self.K2  = K2
        self.lam = lam

    def damped_pinv(self, J):
        """
        Compute damped least squares pseudoinverse.

        Parameters:
            J : matrix to invert (3x6)

        Returns:
            J_pinv : damped pseudoinverse (6x3)
        """
        return J.T @ np.linalg.inv(J @ J.T + self.lam**2 * np.eye(J.shape[0]))

    def null_space(self, J, J_pinv):
        """
        Compute null space projection matrix.

        Parameters:
            J      : Jacobian matrix (3x6)
            J_pinv : damped pseudoinverse of J (6x3)

        Returns:
            N : null space projector (6x6)
        """
        return np.eye(J.shape[1]) - J_pinv @ J

    def step(self, q, p_rcm_target, p_tip_target, dt):
        """
        Compute joint velocities for one timestep.

        Parameters:
            q             : current joint angles (6,)
            p_rcm_target  : fixed trocar point that must not move (3,)
            p_tip_target  : where the tip should be at this timestep (3,)
            dt            : timestep in seconds

        Returns:
            q_new  : updated joint angles (6,)
            e_rcm  : trocar error vector (3,)
            e_tip  : tip error vector (3,)
        """
        q = np.asarray(q, dtype=float)

        # 1. Forward kinematics
        T, transforms = forward_kinematics(q)
        T_wrist = transforms[5]
        p_trocar, p_tip = get_tool_points(T_wrist)

        # 2. Errors
        e_rcm = p_rcm_target - p_trocar
        e_tip = p_tip_target - p_tip

        # 3. Jacobians
        J_rcm = compute_jacobian(q, transforms, p_target=p_trocar)
        J_tip = compute_jacobian(q, transforms, p_target=p_tip)

        # 4. Damped pseudoinverses
        J_rcm_pinv = self.damped_pinv(J_rcm)
        J_tip_pinv = self.damped_pinv(J_tip)

        # 5. Null space projector
        N = self.null_space(J_rcm, J_rcm_pinv)

        # 6. Primary task — drive RCM error to zero
        dq_primary = J_rcm_pinv @ (self.K1 * e_rcm)

        # 7. Secondary task — tip tracking + joint limit avoidance in null space
        dq_tip = J_tip_pinv @ (self.K2 * e_tip)

        # Joint limit avoidance gradient
        from robot.kinematics import joint_limit_gradient
        dq_limits = joint_limit_gradient(q)

        # Project both through null space
        dq_secondary = N @ (dq_tip + 0.1 * dq_limits)

        # 8. Combined joint velocity
        dq = dq_primary + dq_secondary

        # 9. Integrate
        q_new = q + dq * dt

        return q_new, e_rcm, e_tip


class NaiveController:
    """
    Controller A — naive implementation.
    Plain pseudoinverse, no damping, no null space, no drift correction.
    This is the baseline that demonstrates why the full controller is necessary.
    """
    def __init__(self, K=5.0):
        """
        Parameters:
            K : single gain for tip tracking
        """
        self.K = K

    def step(self, q, p_rcm_target, p_tip_target, dt):
        """
        Compute joint velocities using plain pseudoinverse.
        No null space — directly solves for tip tracking only.
        No damping — will blow up near singularities.
        No drift correction.

        Parameters:
            q             : current joint angles (6,)
            p_rcm_target  : trocar target (tracked but not enforced)
            p_tip_target  : where tip should be (3,)
            dt            : timestep in seconds

        Returns:
            q_new  : updated joint angles (6,)
            e_rcm  : trocar error vector (3,)
            e_tip  : tip error vector (3,)
        """
        q = np.asarray(q, dtype=float)

        # Forward kinematics
        T, transforms = forward_kinematics(q)
        T_wrist = transforms[5]
        p_trocar, p_tip = get_tool_points(T_wrist)

        # Errors
        e_rcm = p_rcm_target - p_trocar
        e_tip = p_tip_target - p_tip

        # Tip Jacobian only — no RCM Jacobian
        J_tip = compute_jacobian(q, transforms, p_target=p_tip)

        # Plain pseudoinverse — no damping
        J_tip_pinv = np.linalg.pinv(J_tip)

        # Direct tip tracking — no null space
        dq = J_tip_pinv @ (self.K * e_tip)

        # Integrate
        q_new = q + dq * dt

        return q_new, e_rcm, e_tip

