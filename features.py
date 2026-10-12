import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# The 4 landmarks of each finger, from the base to the tip
FINGER_CHAINS = [
    [1, 2, 3, 4],      # thumb
    [5, 6, 7, 8],      # index
    [9, 10, 11, 12],   # middle
    [13, 14, 15, 16],  # ring
    [17, 18, 19, 20],  # pinky
]

# Direction of each finger (base -> tip), used to measure how spread apart neighbouring fingers are
FINGER_DIRECTIONS = [
    (INDEX_MCP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_TIP),
    (RING_MCP, RING_TIP),
    (PINKY_MCP, PINKY_TIP),
]

N_GEOMETRIC_FEATURES = 10 + 4 + 10 + 3 + 1  # see extract_features


def normalize(sample):
    """
    Transform the array into a 21:3 matrix and normalize it using the wrist as the origin. This is with the objective
    of maintaining consistency in the data while training the model and to avoid that the position of the hand
    affects the prediction
    :param
    sample: np.ndarray
    Array of size 63 with the (x,y,z) coordinates of the 21 mediapipe landmarks in the hand
    :return:
    np.ndarray
    Array of size 63 with the normalized coordinates of the 21 mediapipe landmarks in the hand
    """
    pts = sample.reshape(21, 3)
    pts = pts - pts[0]

    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    return pts.flatten()


def angle_between(v1, v2):
    """
    Computes the angle between two vectors in radians (0 = same direction, pi = opposite directions)
    :param
    v1: np.ndarray
    First vector
    v2: np.ndarray
    Second vector
    :return:
    float
    Angle between the two vectors in radians
    """
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def extract_features(sample):
    """
    Builds the full feature vector of one hand: the 63 normalized coordinates plus geometric features that describe
    the relationships between fingers directly. The raw coordinates force the model to discover these relationships
    by itself, which is hard for small differences such as two fingers being together (U) or apart (V).
    All distances are divided by the hand size (wrist to middle finger base), so they do not depend on how big the
    hand is or how far it is from the camera
    Geometric features (28):
    - 10 distances between every pair of fingertips
    - 4 distances from the thumb tip to the base and middle joint of the index and middle fingers
    - 10 bending angles, two per finger
    - 3 spread angles between neighbouring fingers (index-middle, middle-ring, ring-pinky)
    - 1 crossing value: negative when the index and middle fingers are crossed
    :param
    sample: np.ndarray
    Array of size 63 with the raw (x,y,z) coordinates of the 21 mediapipe landmarks in the hand
    :return:
    np.ndarray
    Array of size 63 + 28 = 91 with the normalized coordinates followed by the geometric features
    """
    coords = normalize(sample)
    pts = coords.reshape(21, 3)

    hand_size = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])
    if hand_size == 0:
        hand_size = 1.0

    geometric = []

    # 1. Distances between every pair of fingertips (5 tips -> 10 pairs)
    for a in range(len(FINGERTIPS)):
        for b in range(a + 1, len(FINGERTIPS)):
            geometric.append(np.linalg.norm(pts[FINGERTIPS[a]] - pts[FINGERTIPS[b]]) / hand_size)

    # 2. Where the thumb tip is, relative to the index and middle fingers
    for joint in [INDEX_MCP, INDEX_PIP, MIDDLE_MCP, MIDDLE_PIP]:
        geometric.append(np.linalg.norm(pts[THUMB_TIP] - pts[joint]) / hand_size)

    # 3. Bending angles: angle at the 2nd and 3rd landmark of each finger (straight finger = pi)
    for chain in FINGER_CHAINS:
        for k in range(1, 3):
            prev_pt, joint, next_pt = pts[chain[k - 1]], pts[chain[k]], pts[chain[k + 1]]
            geometric.append(angle_between(prev_pt - joint, next_pt - joint))

    # 4. Spread between neighbouring fingers (0 = parallel, bigger = more open)
    for (base_a, tip_a), (base_b, tip_b) in zip(FINGER_DIRECTIONS, FINGER_DIRECTIONS[1:]):
        geometric.append(angle_between(pts[tip_a] - pts[base_a], pts[tip_b] - pts[base_b]))

    # 5. Crossing of index and middle fingers. The axis goes from the middle finger base to the index finger base.
    # Normally the index tip is on the same side as the index base (positive value); if the fingers are crossed,
    # the tips swap sides and the value becomes negative. It works for both hands because it uses the hand's own axis
    axis = pts[INDEX_MCP] - pts[MIDDLE_MCP]
    tips = pts[INDEX_TIP] - pts[MIDDLE_TIP]
    geometric.append(np.dot(tips, axis) / (np.dot(axis, axis) + 1e-8))

    return np.concatenate([coords, np.array(geometric)])
