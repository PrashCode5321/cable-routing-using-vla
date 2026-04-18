import numpy as np
import cv2, time
from xarm.wrapper import XArmAPI
from utils import presets
from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from scipy.spatial.transform import Rotation
from detector import BracketDetector
from typing import List

SPEED = 100  # conservative speed for precision work

## TODO: C CLIP - IF TAG - EE POSE = 180 --> TREAT IT LIKE 0 (ALIGNED)

class ActionPlanner:
    def __init__(self, camera_pose: np.array):
        self.arm = XArmAPI(presets.ROBOT_IP)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_tcp_offset([0, 0, presets.GRIPPER_LENGTH, 0, 0, 0])
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.move_gohome(speed=100, wait=True)
        self.t_robot_cam = np.linalg.inv(camera_pose)
        self.config = {
            "y_clip": {"X_OFFSET": -42.0, "Y_OFFSET": 42.0, "Z_SAFE": 60.0},
            "c_clip": {"X_OFFSET": 42.0, "Y_OFFSET": 42.0, "Z_SAFE": 10.0},
            "round_clip": {"RADIUS": 45.0, "Z_SAFE": 60.0}
        }
        print("robot initialized")

    def shutdown(self):
        self.arm.open_lite6_gripper(sync=True)
        self.arm.set_pause_time(sltime=0.2, wait=True)
        _, safe = self.arm.get_position(is_radian=False)
        self.arm.set_position(
            x=safe[0], y=safe[1], z=safe[2]+20, 
            roll=safe[3], pitch=safe[4], yaw=safe[5],
            is_radian=False, speed=100, wait=True
        )
        self.arm.move_gohome(speed=100, wait=True)
        self.arm.close_lite6_gripper(sync=True)
        self.arm.set_pause_time(sltime=0.2, wait=True)
        self.arm.stop_lite6_gripper(sync=True)
        time.sleep(1)
        self.arm.disconnect()
    
    def wrap(self, angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    def equivalent_yaw(self, yaw: float) -> float:
        result = self.arm.get_position(is_radian=False)
        if result[0] != 0:
            return self.wrap(yaw)
        current_yaw = result[1][5]
        candidates = [self.wrap(yaw), self.wrap(yaw + 180.0)]
        return min(candidates, key=lambda x: abs(self.wrap(x - current_yaw)))

    def pose_to_command(self, pose: np.array) -> dict:
        roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        # yaw = self.equivalent_yaw(yaw)
        x_m, y_m, z_m, _ = pose[:, 3].flatten()
        return {"x": x_m * 1000, "y": y_m * 1000, "z": z_m * 1000,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "is_radian": False, "speed": SPEED, "wait": True,
        }

    def execute_plan(self, plan: List[dict]) -> None:
        for step in plan:
            if step.get("log"):
                print(step["log"])
                del step["log"]
            print(step)
            ret = self.arm.set_position(**step)
            if ret != 0:
                raise RuntimeError("Motion failed")
        self.arm.set_pause_time(sltime=0.5, wait=True)
        print("Motion Completed")

    def r_clip_plan(self, clip_pose: np.ndarray) -> List[dict]:
        round_clip_config = self.config["round_clip"]
        RADIUS, Z_SAFE = round_clip_config.values()
        motion_plan = []

        # arc positions
        offsets = [
            np.array([-RADIUS, 0, 0]),
            np.array([0, RADIUS, 0]),
            np.array([RADIUS, 0, 0]),
            np.array([0, -RADIUS, 0]),
            np.array([-RADIUS, 0, 0]),
            np.array([0, RADIUS, 0]),
        ]
        
        intial = self.pose_to_command(pose=self.t_robot_cam @ clip_pose)
        intial["z"] += Z_SAFE
        intial["log"] = "[1/3] On top of bracket …"
        motion_plan.append(intial)

        for offset in offsets:
            t_tag_waypoint = np.eye(4)
            t_tag_waypoint[:3, 3] = offset / 1000
            params = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
            motion_plan.append(params)
        
        intial["log"] = "[3/3] Returning to initial position …"
        motion_plan.append(intial)
        return motion_plan  

    def c_clip_plan(self, clip_pose: np.ndarray) -> List[dict]:
        c_clip_config = self.config["c_clip"]
        X_OFFSET, Y_OFFSET, Z_SAFE = c_clip_config.values()
        motion_plan = []

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = -X_OFFSET / 1000  # along tag's local X
        t_tag_waypoint[1, 3] = Y_OFFSET / 1000
        config = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config["z"] -= Z_SAFE
        config["log"] = "[1/3] Approach to clip opening …"
        motion_plan.append(config)

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = X_OFFSET / 1000  # along tag's local X
        t_tag_waypoint[1, 3] = Y_OFFSET / 1000
        config2 = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config2["z"] -= Z_SAFE
        config2["log"] = "[2/3] Pushing forward along clip …"
        motion_plan.append(config2)

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = X_OFFSET / 1000  # along tag's local X
        config3 = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config3["z"] -= Z_SAFE
        config3["log"] = "[3/3] Seating the wire …"
        motion_plan.append(config3)

        return motion_plan

    def y_clip_plan(self, clip_pose: np.ndarray) -> List[dict]:
        y_clip_config = self.config["y_clip"]
        X_OFFSET, Y_OFFSET, Z_SAFE = y_clip_config.values()
        motion_plan = []

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = X_OFFSET / 1000  # along tag's local X
        t_tag_waypoint[1, 3] = Y_OFFSET / 1000
        config = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config["z"] += Z_SAFE
        config["log"] = "[1/4] Sliding forward to clip opening …"
        motion_plan.append(config)

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = -X_OFFSET / 1000  # along tag's local X
        t_tag_waypoint[1, 3] = Y_OFFSET / 1000
        config2 = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config2["z"] += Z_SAFE
        config2["log"] = "[2/4] Lowering wire into clip channel …"
        motion_plan.append(config2)

        config3 = config2.copy()
        config3["z"] -= Z_SAFE/2
        config3["log"] = "[3/4] Pushing forward to seat wire …"
        motion_plan.append(config3)

        t_tag_waypoint = np.eye(4)
        t_tag_waypoint[0, 3] = -2*X_OFFSET / 1000  # along tag's local X
        t_tag_waypoint[1, 3] = Y_OFFSET / 1000
        config4 = self.pose_to_command(pose=self.t_robot_cam @ clip_pose @ t_tag_waypoint)
        config4["z"] += Z_SAFE/2
        config4["log"] = "[4/4] Lowering wire into clip channel …"
        motion_plan.append(config4)
        
        return motion_plan

def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    cv_image = zed.image
    detector = BracketDetector(
        observation=cv_image,
        intrinsic=camera_intrinsic,
    )
    planner = ActionPlanner(camera_pose=detector.camera_pose)
    try:
        results = detector.identify_april_tag_ids()
        if not results:
            raise Exception("No objects found")
        
        id, t_cam_clip = results[0]
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_clip)
        title = f"Verifying Tag Pose (ID={id})"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280, 720)
        cv2.imshow(title, cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord("k"):
            print("Aborted by user.")
            return

        motion_plan = planner.y_clip_plan(clip_pose=t_cam_clip)
        planner.arm.close_lite6_gripper(sync=True)
        planner.execute_plan(motion_plan)
    except Exception as e:
        raise e
    finally:
        planner.shutdown()
        zed.close()


if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", np.round(time.time() - start, 2), "s")