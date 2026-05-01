import warnings
warnings.filterwarnings("ignore")

from utils.zed_camera import ZedCamera
from xarm.wrapper import XArmAPI
from utils import presets
from PIL import Image
import requests
import io
import numpy as np
import h5py, time, cv2

robot_ip  = presets.ROBOT_IP
GRIPPER_LENGTH = 0.067 * 1000

def main():
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    zed = ZedCamera()
    try:
        while True:
            # Grab image input & format prompt
            TASK_INSTRUCTION = "route the cable through bracket"
            prompt = f"In: What action should the robot take to {TASK_INSTRUCTION}?\nOut:"

            cv_image = zed.image
            cv_image = cv_image.astype(np.uint8, copy=False)
            cv_image = cv2.resize(cv_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image: Image.Image = Image.fromarray(cv_image)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            files = {'image': ('image.png', img_bytes, 'image/png')}
            data = {'instruction': prompt}
            response = requests.post(
                f"{presets.API_URL}/predict-action",
                files=files,
                data=data
            )
            result = response.json()
            if not result.get('success'):
                raise ValueError(f"API Error: {result.get('error', 'Unknown error')}")
            action = result['data']['action']

            # t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
            dx, dy, dz, droll, dpitch, dyaw, gripper = action

            # meters → mm
            dx *= 1000.0
            dy *= 1000.0
            dz *= 1000.0

            # rotations: KEEP as radians (DO NOT scale arbitrarily)
            # xArm expects degrees, so convert once properly
            droll  = np.degrees(droll)
            dpitch = np.degrees(dpitch)
            dyaw   = np.degrees(dyaw)
            _, pose = arm.get_position(is_radian=False)
            x, y, z, roll, pitch, yaw = pose

            new_x = x + dx + 5
            new_y = y + dy + 5
            new_z = z + dz + 5

            new_roll  = roll + droll
            new_pitch = pitch + dpitch
            new_yaw   = yaw + dyaw
            print("Current: ", x, y, z, roll, pitch, yaw)
            print("New: ", new_x, new_y, new_z, new_roll, new_pitch, new_yaw)
            arm.set_position(
                x=new_x,
                y=new_y,
                z=new_z,
                roll=new_roll,
                pitch=new_pitch,
                yaw=new_yaw,
                speed=80,
                wait=True
            )
            time.sleep(1)
    except Exception as e:
        raise e
    finally:
        arm.move_gohome(wait=True)
        arm.disconnect()
        zed.close()
        arm.open_lite6_gripper(sync=True)
        arm.set_pause_time(sltime=0.2, wait=True)
        _, safe = arm.get_position(is_radian=False)
        arm.set_position(
            x=safe[0], y=safe[1], z=safe[2]+20, 
            roll=safe[3], pitch=safe[4], yaw=safe[5],
            is_radian=False, speed=presets.SPEED, wait=True
        )
        arm.move_gohome(speed=presets.SPEED, wait=True)
        arm.close_lite6_gripper(sync=True)
        arm.set_pause_time(sltime=0.2, wait=True)
        arm.stop_lite6_gripper(sync=True)
        time.sleep(1)
        arm.disconnect()
        print("robot shutdown complete")

if __name__ == "__main__":
    main()