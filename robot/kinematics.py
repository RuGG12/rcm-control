import numpy as np


def dh_transform(a, d, alpha, theta):
    # Standard DH transform for one joint.
    # a = link length, d = link offset, alpha = link twist, theta = joint angle
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca,  st*sa,  a*ct],
        [st,  ct*ca, -ct*sa,  a*st],
        [0,   sa,     ca,     d   ],
        [0,   0,      0,      1   ]
    ])

    return T


# UR5 DH parameters from Universal Robots datasheet
# Each row: [a, d, alpha, theta_offset]
# theta_offset is added to q[i] at runtime (all zero for UR5)
UR5_DH = [
    #    a        d        alpha     theta_offset
    [0.0,      0.08916,  np.pi/2,   0.0],   # joint 1
    [-0.42500, 0.0,      0.0,       0.0],   # joint 2
    [-0.39225, 0.0,      0.0,       0.0],   # joint 3
    [0.0,      0.10915,  np.pi/2,   0.0],   # joint 4
    [0.0,      0.09465, -np.pi/2,   0.0],   # joint 5
    [0.0,      0.0823,   0.0,       0.0],   # joint 6
]

# UR5 joint limits in radians (from UR5 spec)
# Joint 3 is more restricted than the others
UR5_JOINT_LIMITS = {
    'min': np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
    'max': np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])
}

# Tool geometry — change these if the physical setup changes
TOOL_LENGTH  = 0.20   # wrist to tip, meters
TROCAR_DEPTH = 0.10   # wrist to trocar point, meters


def joint_limit_gradient(q):
    # Gradient of a quadratic potential centered at the midpoint of each joint range.
    # Returns a vector that points toward the center — use this in the null space
    # to gently push joints away from their limits.
    q_min = UR5_JOINT_LIMITS['min']
    q_max = UR5_JOINT_LIMITS['max']
    q_mid = 0.5 * (q_max + q_min)
    q_range = q_max - q_min

    grad = (q - q_mid) / (q_range ** 2)
    return -grad


def forward_kinematics(q):
    # Chain the six DH transforms to get the full base-to-wrist transform.
    # Returns the final 4x4 matrix and a list of all six intermediate frames.
    q = np.asarray(q, dtype=float)

    T = np.eye(4)
    transforms = []

    for i in range(6):
        a, d, alpha, theta_offset = UR5_DH[i]
        theta = q[i] + theta_offset
        Ti = dh_transform(a, d, alpha, theta)
        T = T @ Ti
        transforms.append(T.copy())

    return T, transforms


def get_position(T):
    return T[:3, 3]


def get_rotation(T):
    return T[:3, :3]


def compute_jacobian(q, transforms, p_target=None):
    # Geometric Jacobian for any point on the robot.
    # Each column is z_{i-1} x (p_target - p_{i-1}) — standard revolute joint formula.
    # If p_target is not given, defaults to the wrist position.
    if p_target is None:
        p_target = transforms[5][:3, 3]

    J = np.zeros((3, 6))

    for i in range(6):
        if i == 0:
            z = np.array([0, 0, 1])
            p = np.array([0, 0, 0])
        else:
            z = transforms[i-1][:3, 2]
            p = transforms[i-1][:3, 3]

        J[:, i] = np.cross(z, p_target - p)

    return J


def get_tool_points(T_wrist, tool_length=TOOL_LENGTH, trocar_depth=TROCAR_DEPTH):
    # The tool shaft runs along the wrist Z axis.
    # Project out by trocar_depth to get the trocar point, tool_length to get the tip.
    p_wrist = T_wrist[:3, 3]
    z_hat   = T_wrist[:3, 2]

    p_trocar = p_wrist + trocar_depth * z_hat
    p_tip    = p_wrist + tool_length  * z_hat

    return p_trocar, p_tip