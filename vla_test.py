from transformers import AutoModelForVision2Seq, AutoProcessor
from utils.zed_camera import ZedCamera
from utils.workspace_check import get_transform_camera_robot
from xarm.wrapper import XArmAPI
from utils import presets
from PIL import Image
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import checkpoint6


torch.cuda.empty_cache()
# zed = ZedCamera()
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
    camera_intrinsic = zed.camera_intrinsic
    try:
        cv_image = zed.image

        # Load Processor & VLA
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cpu")

        # Grab image input & format prompt
        image: Image.Image = Image.fromarray(cv_image)
        prompt = "In: What action should the robot take to pick  the red cube?\nOut:"

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        # Execute...

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
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

        new_x = x + dx
        new_y = y + dy
        new_z = z + dz

        new_roll  = roll + droll
        new_pitch = pitch + dpitch
        new_yaw   = yaw + dyaw
        print("Ground: ", checkpoint6.main(zed=zed))
        print("Current: ", x, y, z, roll, pitch, yaw)
        print("New: ", new_x, new_y, new_z, new_roll, new_pitch, new_yaw)
        print("RAW ACTION:", action)
        print("MAX ABS:", np.max(np.abs(action[:6])))
        arm.set_position(
            x=new_x,
            y=new_y,
            z=new_z,
            roll=new_roll,
            pitch=new_pitch,
            yaw=new_yaw,
            speed=50,
            mvacc=200,
            wait=False
        )
        # robot.act(action, ...)
    except Exception as e:
        raise e
    finally:
        arm.move_gohome(wait=True)
        arm.disconnect()
        zed.close()

if __name__ == "__main__":
    main()