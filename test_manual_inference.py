import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
import h5py
import cv2
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
CHECKPOINT_PATH = "/home/prashant-rao/Documents/PythonWorkspace/5551/openvla/runs/openvla-7b+my_robot_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
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

def main(image: np.ndarray) -> List[float]:
    
    try:
        # Grab image input & format prompt
        TASK_INSTRUCTION = "route the cable through bracket"
        prompt = f"In: What action should the robot take to {TASK_INSTRUCTION}?\nOut:"

        # Prepare inputs
        image = Image.fromarray(image.astype(np.uint8))
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
        
        return action
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    finally:
        pass


def load_hdf5_episode(filepath: str) -> Dict[str, List[float] | List[str] | List[np.ndarray]]:
    """Load episode data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        print(len(f), print(list(f.keys())))
        return {
            "images":       f["observations/images"][:],
            "ee_poses":     f["observations/ee_poses"][:],
            "joint_states": f["observations/joint_states"][:],
            "actions":      f["ee_actions"][:],
        }
    

if __name__ == "__main__":
    frames = load_hdf5_episode("/home/prashant-rao/Documents/PythonWorkspace/5551/episodes/episode_0000.hdf5")
    for frame, ee in zip(frames["images"], frames["ee_poses"]):
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        action = main(frame)
        print(action, ee)
