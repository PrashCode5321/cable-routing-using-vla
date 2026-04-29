import cv2, numpy, open3d
from utils import logger
import logging

logger = logging.getLogger("VLA")

def draw_pose_axes(image, camera_intrinsic, pose, size=0.1):
    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    # origin and 3 unit vector of the frame
    frame_points = numpy.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]).reshape(-1,3) * size

    ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    ipoints = numpy.round(ipoints).astype(int)

    origin = tuple(ipoints[0].ravel())
    unit_x = tuple(ipoints[1].ravel())
    unit_y = tuple(ipoints[2].ravel())
    unit_z = tuple(ipoints[3].ravel())

    cv2.line(image, origin, unit_x, (0,0,255), 2)
    cv2.line(image, origin, unit_y, (0,255,0), 2)
    cv2.line(image, origin, unit_z, (255,0,0), 2)

    # if t_robot:
    #     rob_tvec = t_robot[:3, 3]
    #     text = f"({rob_tvec[0]:.3f}, {rob_tvec[1]:.3f}, {rob_tvec[2]:.3f})"
    #     cv2.putText(image, text, (origin[0] + 5, origin[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1, cv2.LINE_AA)

def get_workspace_mask(bgr_image):
    """
    Detect the white arena poster and return a binary mask of its region.

    Returns
    -------
    workspace_mask : numpy.ndarray (HxW, uint8)
        255 inside the poster region, 0 outside.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # White = low saturation, high value
    white_mask = cv2.inRange(hsv,
                             numpy.array([  0,   0, 180]),
                             numpy.array([180,  50, 255]))

    # Clean up noise
    kernel = numpy.ones((15, 15), numpy.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)

    # Find the largest contour — that's the poster
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("[workspace] No white region found — using full image.")
        return numpy.full(bgr_image.shape[:2], 255, dtype=numpy.uint8)

    largest = max(contours, key=cv2.contourArea)

    workspace_mask = numpy.zeros(bgr_image.shape[:2], dtype=numpy.uint8)
    cv2.drawContours(workspace_mask, [largest], -1, color=255, thickness=cv2.FILLED)

    return workspace_mask

