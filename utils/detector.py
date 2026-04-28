from utils.vis_utils import draw_pose_axes, get_workspace_mask
from utils.workspace_check import get_transform_camera_robot
from utils.zed_camera import ZedCamera
from pupil_apriltags import Detector
from typing import Dict, List, Tuple
import numpy as np
import cv2

COLOR_HSV_RANGES = {
    'red':   [((  0, 100,  60), ( 10, 255, 255)),
                ((160, 100,  60), (180, 255, 255))],
    'green': [((40,   60,  40), ( 90, 255, 255))],
    'blue':  [((100,  80,  40), (140, 255, 255))],
}

class BracketDetector:
    def __init__(
            self, 
            observation: np.array, 
            intrinsic: np.array, 
            tag_size: float= 0.0207, 
            family: str="tag36h11"
        ) -> None:
        self.tag_size = tag_size
        self.camera_intrinsic = intrinsic
        self.image = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)

        # april
        self.detector = Detector(families=family)
        self.camera_pose = get_transform_camera_robot(self.image, self.camera_intrinsic)

    def identify_april_tag_ids(self) -> List[Tuple[int, np.array]]:
        """
        This function identifies tag ID, color and pose of the cube.  
        Output:  
            `List[Tuple[int, np.array]]` - list of detected tags with their ID and pose in camera frame
        """
        camera_params = [
            self.camera_intrinsic[0, 0], 
            self.camera_intrinsic[1, 1], 
            self.camera_intrinsic[0, 2], 
            self.camera_intrinsic[1, 2]
        ]
        tags = self.detector.detect(
            self.image, 
            estimate_tag_pose=True, 
            camera_params=camera_params, 
            tag_size=self.tag_size
        )

        detected_tags = []
        for tag in tags:
            # 1. Identify the tag ID of the cube
            if tag.tag_id <= 3:
                continue
            
            # 2. Identify the pose of the cube
            t_cam_cube = np.eye(4)
            t_cam_cube[:3, :3] = tag.pose_R
            t_cam_cube[:3, 3] = tag.pose_t.flatten()
            detected_tags.append((tag.tag_id, t_cam_cube))
        return detected_tags


if __name__ == "__main__":
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    try:
        # Get Observation
        cv_image = zed.image

        # Initialize Tag Pose Detector
        detector = BracketDetector(
            observation=cv_image,
            intrinsic=camera_intrinsic,
        )
        results = detector.identify_april_tag_ids()
        
        if not results:
            raise("No matching object found for ID")
        
        cv2.namedWindow('Verifying Tag Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Tag Pose', 1280, 720)
        for result in results:
            _, t_cam_cube = result            
            draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        
        cv2.imshow('Verifying Tag Pose', cv_image)
        key = cv2.waitKey(0)
    
        if key == ord('k'):
            cv2.destroyAllWindows()
            
    finally:
        # Close ZED Camera
        zed.close()
