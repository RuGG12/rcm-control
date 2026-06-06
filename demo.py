"""
demo.py

Runs the RCM controller through several tests in sequence:

  Phase 1 - kinematics_check()
      FK and Jacobian sanity checks at known configurations.

  Phase 2 - controller_step_test()
      Single step smoke test. Checks that one controller step produces
      near-zero RCM error.

  Phase 3 - simulation_run()
      500 steps with the circular tip trajectory. Prints RCM/tip error stats.

  Phase 4 - controller_comparison()
      RCMController vs NaiveController on the same trajectory.
      Saves rcm_comparison.png.

  Phase 5 - lambda_sweep_phase()
      Sweeps lambda across [0.001, 0.01, 0.05, 0.1, 0.5].
      Saves lambda_sweep.png.

  Phase 6 - animation_phase()
      Runs a fresh simulation and saves rcm_animation.gif.

  Phase 7 - joint_limit_check()
      Verifies joint angles stay within UR5 limits across 500 steps.
      Prints the metrics summary.

Usage:
    python demo.py
"""

import numpy as np
from robot.kinematics import forward_kinematics, compute_jacobian, get_tool_points
from robot.controller import RCMController, NaiveController
from simulation.sim_loop import run_simulation, sweep_lambda, print_metrics
from visualization.visualize import plot_rcm_comparison, plot_lambda_sweep, animate_simulation


def get_initial_state():
    # Start from the UR5 zero configuration and fix the trocar where it sits.
    q0 = np.zeros(6)
    T, transforms = forward_kinematics(q0)
    p_trocar, _ = get_tool_points(transforms[5])
    return q0, p_trocar.copy()


def kinematics_check():
    print("=" * 55)
    print("  PHASE 1 - Kinematics Check")
    print("=" * 55)

    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    print(f"\nFK (all zeros) end effector: {np.round(T[:3, 3], 4)}")

    q_down = np.array([0, -np.pi/2, 0, 0, 0, 0])
    T2, _ = forward_kinematics(q_down)
    print(f"FK (shoulder down) end effector: {np.round(T2[:3, 3], 4)}")

    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    J = compute_jacobian(q, transforms)
    print(f"\nJacobian shape: {J.shape}  (expected: (3, 6))")
    print(f"J row 0: {np.round(J[0], 4)}")

    T_wrist = transforms[5]
    p_trocar, p_tip = get_tool_points(T_wrist)
    print(f"\nWrist : {np.round(T_wrist[:3, 3], 4)}")
    print(f"Trocar: {np.round(p_trocar, 4)}")
    print(f"Tip   : {np.round(p_tip, 4)}")


def controller_step_test():
    print("\n" + "=" * 55)
    print("  PHASE 2 - Controller Step Test")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()
    _, transforms = forward_kinematics(q0)
    _, p_tip = get_tool_points(transforms[5])
    p_tip_target = p_tip + np.array([0.05, 0.0, 0.0])

    ctrl = RCMController(K1=5.0, K2=3.0, lam=0.01)
    q_new, e_rcm, e_tip = ctrl.step(q0, p_rcm_target, p_tip_target, dt=0.01)

    print(f"\nq_new  : {np.round(q_new, 6)}")
    print(f"e_rcm  : {np.round(e_rcm, 6)}  (should be near zero)")
    print(f"|e_rcm|: {np.linalg.norm(e_rcm):.6f} m")
    print(f"|e_tip|: {np.linalg.norm(e_tip):.6f} m")


def simulation_run():
    print("\n" + "=" * 55)
    print("  PHASE 3 - Simulation Run (500 steps)")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()
    history = run_simulation(q0, p_rcm_target=p_rcm_target, dt=0.01, steps=500)

    e_rcm_arr = np.array(history['e_rcm_norm'])
    e_tip_arr  = np.array(history['e_tip_norm'])
    n_corr     = sum(history['drift_corrections'])

    print(f"\nRCM error  mean: {e_rcm_arr.mean()*1000:.4f} mm  max: {e_rcm_arr.max()*1000:.4f} mm")
    print(f"Tip error  mean: {e_tip_arr.mean()*1000:.4f} mm  max: {e_tip_arr.max()*1000:.4f} mm")
    print(f"Drift corrections applied: {n_corr}")

    return history


def controller_comparison():
    print("\n" + "=" * 55)
    print("  PHASE 4 - Controller Comparison")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()

    def run_ctrl(ctrl, q0, p_rcm_target, dt=0.01, steps=500):
        q = q0.copy()
        e_rcm_log, e_tip_log = [], []
        for step in range(steps):
            t = step * dt
            center = p_rcm_target + np.array([0.0, 0.1, 0.0])
            p_tip_target = center + np.array([0.05 * np.cos(t), 0.0, 0.05 * np.sin(t)])
            q, e_rcm, e_tip = ctrl.step(q, p_rcm_target, p_tip_target, dt)
            e_rcm_log.append(np.linalg.norm(e_rcm))
            e_tip_log.append(np.linalg.norm(e_tip))
        return np.array(e_rcm_log), np.array(e_tip_log)

    rcm_errors, tip_errors_rcm = run_ctrl(
        RCMController(K1=5.0, K2=3.0, lam=0.01), q0, p_rcm_target)
    naive_errors, tip_errors_naive = run_ctrl(
        NaiveController(K=5.0), q0, p_rcm_target)

    print(f"\n{'':20s}  {'RCM mean':>10s}  {'RCM max':>10s}  {'Tip mean':>10s}")
    print(f"{'RCMController':20s}  "
          f"{rcm_errors.mean()*1000:10.4f}  "
          f"{rcm_errors.max()*1000:10.4f}  "
          f"{tip_errors_rcm.mean()*1000:10.4f}")
    print(f"{'NaiveController':20s}  "
          f"{naive_errors.mean()*1000:10.4f}  "
          f"{naive_errors.max()*1000:10.4f}  "
          f"{tip_errors_naive.mean()*1000:10.4f}")
    print("  (all values in mm)")

    plot_rcm_comparison(rcm_errors, naive_errors)
    return rcm_errors, naive_errors


def lambda_sweep_phase():
    print("\n" + "=" * 55)
    print("  PHASE 5 - Lambda Sweep")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()
    lambdas = [0.001, 0.01, 0.05, 0.1, 0.5]
    sweep_results = sweep_lambda(q0, p_rcm_target, lambdas)
    plot_lambda_sweep(sweep_results)
    return sweep_results


def animation_phase():
    print("\n" + "=" * 55)
    print("  PHASE 6 - Animation")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()
    history = run_simulation(q0, p_rcm_target=p_rcm_target, dt=0.01, steps=500)
    animate_simulation(history, p_rcm_target, dt=0.01, save_gif=True)


def joint_limit_check():
    print("\n" + "=" * 55)
    print("  PHASE 7 - Joint Limit Check")
    print("=" * 55)

    q0, p_rcm_target = get_initial_state()
    history = run_simulation(q0, p_rcm_target=p_rcm_target, dt=0.01, steps=500)
    q_history = np.array(history['q'])

    q_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
    q_max = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

    print()
    for i in range(6):
        min_val = q_history[:, i].min()
        max_val = q_history[:, i].max()
        status = "OK" if min_val >= q_min[i] and max_val <= q_max[i] else "VIOLATED"
        print(f"  Joint {i+1}: min={min_val:7.3f}  max={max_val:7.3f}  "
              f"limits=[{q_min[i]:.3f}, {q_max[i]:.3f}]  {status}")

    print_metrics(history)


if __name__ == "__main__":
    kinematics_check()
    controller_step_test()
    simulation_run()
    controller_comparison()
    lambda_sweep_phase()
    animation_phase()
    joint_limit_check()