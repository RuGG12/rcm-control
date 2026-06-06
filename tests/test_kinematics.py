import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robot.kinematics import (
    forward_kinematics,
    compute_jacobian,
    get_tool_points,
    dh_transform
)

def test_fk_dimensions():
    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    assert T.shape == (4, 4), "FK output must be 4x4"
    assert len(transforms) == 6, "Must have 6 intermediate transforms"
    print("✓ test_fk_dimensions")

def test_fk_bottom_row():
    q = np.random.uniform(-np.pi, np.pi, 6)
    T, _ = forward_kinematics(q)
    expected = np.array([0, 0, 0, 1])
    assert np.allclose(T[3, :], expected), "Bottom row must be [0,0,0,1]"
    print("✓ test_fk_bottom_row")

def test_jacobian_dimensions():
    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    J = compute_jacobian(q, transforms)
    assert J.shape == (3, 6), "Jacobian must be 3x6"
    print("✓ test_jacobian_dimensions")

def test_null_space_property():
    """
    N = I - J†J
    For any vector v: J @ N @ v should be near zero.
    With damped pseudoinverse this is approximate, not exact.
    """
    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    p_trocar, _ = get_tool_points(transforms[5])
    J = compute_jacobian(q, transforms, p_target=p_trocar)

    lam = 0.01
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
    N = np.eye(6) - J_pinv @ J

    v = np.random.randn(6)
    leakage = np.linalg.norm(J @ N @ v)

    # With damped LS, leakage is small but not zero
    assert leakage < 0.1, f"Null space leakage too large: {leakage:.6f}"
    print(f"✓ test_null_space_property (leakage={leakage:.6f})")

def test_dls_stability():
    """
    DLS pseudoinverse should never produce norm > 1/lambda
    even near singular configurations.
    """
    lam = 0.01
    # Near-singular Jacobian — two identical columns
    J = np.zeros((3, 6))
    J[:, 0] = [1, 0, 0]
    J[:, 1] = [1, 0, 0]  # duplicate — near singular
    J[:, 2] = [0, 1, 0]

    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
    max_gain = np.linalg.norm(J_pinv, ord=2)

    assert max_gain < 1.0 / lam, f"DLS gain too large: {max_gain:.4f}"
    print(f"✓ test_dls_stability (max_gain={max_gain:.4f})")

def test_tool_points_on_shaft():
    """
    Trocar and tip must lie on the tool shaft direction from wrist.
    """
    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    p_trocar, p_tip = get_tool_points(transforms[5])
    p_wrist = transforms[5][:3, 3]
    u_shaft = transforms[5][:3, 2]
    u_shaft = u_shaft / np.linalg.norm(u_shaft)

    # Both points should be collinear with wrist along u_shaft
    v_trocar = p_trocar - p_wrist
    v_tip = p_tip - p_wrist

    # Cross product with u_shaft should be near zero
    assert np.linalg.norm(np.cross(v_trocar, u_shaft)) < 1e-10
    assert np.linalg.norm(np.cross(v_tip, u_shaft)) < 1e-10
    print("✓ test_tool_points_on_shaft")

if __name__ == "__main__":
    test_fk_dimensions()
    test_fk_bottom_row()
    test_jacobian_dimensions()
    test_null_space_property()
    test_dls_stability()
    test_tool_points_on_shaft()
    print("\nAll tests passed.")
