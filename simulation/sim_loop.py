import numpy as np
from robot.kinematics import forward_kinematics, get_tool_points, compute_jacobian
from robot.controller import RCMController


def run_simulation(q0, p_rcm_target, dt=0.01, steps=500,
                   correction_interval=50, drift_threshold=0.0001):
    # Runs the full simulation loop.
    # Tip follows a 5cm radius circle in the XZ plane, 10cm above the trocar.
    # Every correction_interval steps, checks trocar drift and corrects if needed.
    controller = RCMController(K1=5.0, K2=3.0, lam=0.01)
    q = q0.copy()

    history = {
        'e_rcm_norm':        [],
        'e_tip_norm':        [],
        'q':                 [],
        'p_trocar':          [],
        'p_tip':             [],
        'drift_corrections': []
    }

    for step in range(steps):
        t = step * dt
        radius = 0.05
        center = p_rcm_target + np.array([0.0, 0.1, 0.0])
        p_tip_target = center + np.array([
            radius * np.cos(t),
            0.0,
            radius * np.sin(t)
        ])

        q_new, e_rcm, e_tip = controller.step(q, p_rcm_target, p_tip_target, dt)

        # Drift correction: measure actual trocar position after the step
        # and push it back if it's drifted too far
        T, transforms = forward_kinematics(q_new)
        p_trocar, p_tip = get_tool_points(transforms[5])
        e_drift = p_rcm_target - p_trocar

        correction_applied = False
        if step % correction_interval == 0:
            if np.linalg.norm(e_drift) > drift_threshold:
                J_rcm = compute_jacobian(q_new, transforms, p_target=p_trocar)
                J_rcm_pinv = controller.damped_pinv(J_rcm)
                dq_correction = J_rcm_pinv @ (controller.K1 * e_drift)
                q_new = q_new + dq_correction
                correction_applied = True

        history['e_rcm_norm'].append(np.linalg.norm(e_rcm))
        history['e_tip_norm'].append(np.linalg.norm(e_tip))
        history['q'].append(q_new.copy())
        history['p_trocar'].append(p_trocar.copy())
        history['p_tip'].append(p_tip.copy())
        history['drift_corrections'].append(correction_applied)

        q = q_new

    return history


def sweep_lambda(q0, p_rcm_target, lambdas, dt=0.01, steps=500):
    # Run the same simulation for each lambda value and record RCM error.
    # Useful for picking the right damping parameter.
    results = {}

    for lam in lambdas:
        ctrl = RCMController(K1=5.0, K2=3.0, lam=lam)
        q = q0.copy()
        e_rcm_log = []

        for step in range(steps):
            t = step * dt
            center = p_rcm_target + np.array([0.0, 0.1, 0.0])
            p_tip_target = center + np.array([
                0.05 * np.cos(t),
                0.0,
                0.05 * np.sin(t)
            ])
            q, e_rcm, _ = ctrl.step(q, p_rcm_target, p_tip_target, dt)
            e_rcm_log.append(np.linalg.norm(e_rcm))

        results[lam] = np.array(e_rcm_log)
        print(f"lambda={lam:.3f}  mean={np.mean(e_rcm_log)*1000:.4f}mm  "
              f"max={np.max(e_rcm_log)*1000:.4f}mm")

    return results


def print_metrics(history, dt=0.01):
    # Print a summary of simulation results.
    e_rcm = np.array(history['e_rcm_norm']) * 1000  # meters to mm
    e_tip = np.array(history['e_tip_norm']) * 1000
    n_corrections = sum(history['drift_corrections'])
    duration = len(e_rcm) * dt

    print("\n" + "="*55)
    print("  RCM CONTROLLER - SIMULATION RESULTS")
    print("="*55)
    print(f"  Duration        : {duration:.2f} s")
    print(f"  Steps           : {len(e_rcm)}")
    print(f"  Drift corrections: {n_corrections}")
    print("-"*55)
    print(f"  RCM error mean  : {e_rcm.mean():.4f} mm")
    print(f"  RCM error max   : {e_rcm.max():.4f} mm")
    print(f"  RCM error std   : {e_rcm.std():.4f} mm")
    print("-"*55)
    print(f"  Tolerance (lit) : 29.8000 mm")
    print(f"  Target          : 0.5000 mm")
    print(f"  Achieved        : {e_rcm.mean():.4f} mm")
    print(f"  Margin          : {29.8/e_rcm.mean():.0f}x")
    print("="*55)
