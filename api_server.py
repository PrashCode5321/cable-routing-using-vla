from pathlib import Path
base_dir = Path(__file__).parent / "runs"
CHECKPOINT_PATH = next(base_dir.iterdir())
print(f"Absolute path: {CHECKPOINT_PATH}")

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from PIL import Image
import numpy as np
import torch, h5py, time
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import io

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
print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")

vla = AutoModelForVision2Seq.from_pretrained(
    CHECKPOINT_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # local_files_only=True,
)
vla = vla.to("cuda")
print(f"Model loaded. norm_stats keys: {list(vla.norm_stats.keys()) if hasattr(vla, 'norm_stats') else 'None'}")

def main(cv_image: np.ndarray, instruction: str) -> Dict[str, List[float]]:
    try:
        # Grab image input & format prompt
        TASK_INSTRUCTION = instruction
        prompt = f"In: What action should the robot take to {TASK_INSTRUCTION}?\nOut:"

        image: Image.Image = Image.fromarray(cv_image)
        inputs = processor(prompt, image).to("cuda", dtype=torch.float32)

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

        return {"action": [float(i) for i in action]}
    except Exception as e:
        raise e

# Initialize FastAPI app
app = FastAPI(title="Robot Vision-Language Action API")

@app.post("/predict-action")
async def predict_action(image: UploadFile = File(...), instruction: str = Form(...)):
    """
    Predict robot action from image and instruction.
    
    Args:
        image: Image file (multipart/form-data)
        instruction: Task instruction text
        
    Returns:
        JSON with predicted actions
    """
    try:
        # Read image file
        contents = await image.read()
        img_array = np.array(Image.open(io.BytesIO(contents)))
        
        # Get prediction
        result = main(img_array, instruction)
        return JSONResponse({"success": True, "data": result})
    
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "openvla-7b"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23893)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
