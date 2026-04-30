import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from utils.zed_camera import ZedCamera
# from utils.workspace_check import get_transform_camera_robot
from xarm.wrapper import XArmAPI
from utils import presets
from PIL import Image
import numpy as np
import torch, h5py, time

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

torch.cuda.empty_cache()

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor from base model
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", 
    trust_remote_code=True,
)

# processor = AutoProcessor.from_pretrained(
#     CHECKPOINT_PATH,  # Load from your checkpoint for consistency
#     trust_remote_code=True,
#     local_files_only=True
# )

# Load directly from the patched checkpoint (with norm_stats already in config)
CHECKPOINT_PATH = "/home/rob/csci5551/openvla/runs/openvla-7b+my_robot_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")

vla = AutoModelForVision2Seq.from_pretrained(
    CHECKPOINT_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
)
vla = vla.to("cpu")
print(f"Model loaded. norm_stats keys: {list(vla.norm_stats.keys()) if hasattr(vla, 'norm_stats') else 'None'}")

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
            image: Image.Image = Image.fromarray(cv_image)
            inputs = processor(prompt, image).to("cpu", dtype=torch.float32)

            # Debug: print input shapes
            print(f"Input shapes - input_ids: {inputs['input_ids'].shape}, pixel_values: {inputs['pixel_values'].shape}, attention_mask: {inputs['attention_mask'].shape}")
            
            # Manual generation with disable_cache to avoid shape issues
            print("Generating action tokens...")
            action_dim = len(vla.norm_stats["my_robot_dataset"]["action"]["q01"])
            print(f"Action dimension: {action_dim}")

            with torch.no_grad():
                generated_ids = vla.generate(
                    inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=action_dim,
                    do_sample=False,
                    use_cache=False,
                )
            
            # Decode to continuous actions
            print(f"Generated token shape: {generated_ids.shape}")
            action_tokenizer = ActionTokenizer(processor.tokenizer)
            action_tokens = generated_ids[0, inputs["input_ids"].shape[1]:]
            
            print(f"Action tokens: {action_tokens}")
            action = action_tokenizer.decode_token_ids_to_actions(action_tokens.cpu().numpy())
            print(f"Raw action: {action}")
            
            # De-normalize using dataset statistics
            norm_stats = vla.norm_stats["my_robot_dataset"]["action"]
            action = action * np.array(norm_stats["std"]) + np.array(norm_stats["mean"])
            print(f"Denormalized action: {action}")

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