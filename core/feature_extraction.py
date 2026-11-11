
import cv2
import numpy as np


# Facial landmark indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_pixel_coordinates(landmark, width, height):
    return np.array([
        landmark.x * width,
        landmark.y * height
    ], dtype=np.float64)


# ==========================================
# EYE ASPECT RATIO
# ==========================================

def calculate_ear(landmarks, eye_indices, width, height):

    points = [
        get_pixel_coordinates(landmarks[i], width, height)
        for i in eye_indices
    ]

    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    if horizontal < 1e-6:
        return 0.0

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


# ==========================================
# GAZE ESTIMATION
# ==========================================

def calculate_gaze_x(landmarks, width, height):

    try:
        left_iris = get_pixel_coordinates(
            landmarks[LEFT_IRIS_CENTER], width, height
        )

        right_iris = get_pixel_coordinates(
            landmarks[RIGHT_IRIS_CENTER], width, height
        )

        left_corners = [
            get_pixel_coordinates(landmarks[i], width, height)
            for i in LEFT_EYE_CORNERS
        ]

        right_corners = [
            get_pixel_coordinates(landmarks[i], width, height)
            for i in RIGHT_EYE_CORNERS
        ]

        left_min = min(p[0] for p in left_corners)
        left_max = max(p[0] for p in left_corners)

        right_min = min(p[0] for p in right_corners)
        right_max = max(p[0] for p in right_corners)

        if left_max - left_min < 1e-6:
            return None

        if right_max - right_min < 1e-6:
            return None

        left_ratio = (
            left_iris[0] - left_min
        ) / (left_max - left_min)

        right_ratio = (
            right_iris[0] - right_min
        ) / (right_max - right_min)

        return float((left_ratio + right_ratio) / 2.0)

    except (IndexError, ValueError, TypeError):
        return None


# ==========================================
# HEAD POSE ESTIMATION
# ==========================================

def estimate_head_pose(landmarks, width, height):

    indices = [1, 152, 33, 263, 61, 291]

    image_points = np.array([
        get_pixel_coordinates(landmarks[i], width, height)
        for i in indices
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        pitch, yaw, roll = map(float, angles)

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }

    except (cv2.error, ValueError, TypeError):
        return None


# ==========================================
# ANGLE DIFFERENCE
# ==========================================

def angle_difference(current, baseline):

    return float(
        (current - baseline + 180) % 360 - 180
    )
