# OpenVLA Custom Dataset Integration Changes

## Summary
Documentation of all changes made to the OpenVLA codebase to support the custom `my_robot_dataset` for finetuning and inference.

---

## File Changes

### 1. **mixtures.py**
**Location:** `openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py`

**Change:** Added custom dataset to the OXE_NAMED_MIXTURES registry

```python
# Added line in OXE_NAMED_MIXTURES dictionary:
"my_robot_dataset": [("my_robot_dataset", 1.0)],
```

**Purpose:** Registers the custom dataset as a selectable mixture for finetuning via the `--dataset_name my_robot_dataset` flag.

---

### 2. **configs.py**
**Location:** `openvla/prismatic/vla/datasets/rlds/oxe/configs.py`

**Change:** Added dataset configuration for `my_robot_dataset` in `OXE_DATASET_CONFIGS`:
```python
# Added configuration dict for my_robot_dataset with:
# - image_obs_keys: primary, secondary, wrist camera configurations
# - state_obs_keys: proprioceptive state (joint angles + gripper)
# - state_encoding: StateEncoding.JOINT (7-dim + gripper)
# - action_encoding: ActionEncoding.EEF_WORLD_ALIGNED (7-dim actions)
# - language_instruction_key: "step_language_instruction"
"my_robot_dataset": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}
```

**Purpose:** Defines how observations and actions are structured for your dataset during loading.

---

### 3. **transforms.py**
**Location:** `openvla/prismatic/vla/datasets/rlds/oxe/transforms.py`

**Changes:** 
1. Added transform function for dataset standardization
2. Registered transform in OXE_STANDARDIZATION_TRANSFORMS registry

```python
# Added function:
def my_robot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Standardizes trajectory format to OpenVLA format
    # Extracts images, states, actions, and language instructions
    ...

# Added to registry:
OXE_STANDARDIZATION_TRANSFORMS = {
    "my_robot_dataset": my_robot_dataset_transform,
    # ... other datasets ...
}
```

**Purpose:** Ensures your HDF5 data is converted to the standard OpenVLA trajectory format during training.

---

### 4. **finetune.py** (only for fine-tuning)
**Location:** `openvla/vla-scripts/finetune.py`

#### Change A: Register Dataset Statistics in Config (Device Agnostic)

**Location:** Around line 320-325 (during LoRA merge and checkpoint save)

```python
# OLD
if cfg.use_lora:
    base_vla = AutoModelForVision2Seq.from_pretrained(...)
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()
    
    if distributed_state.is_main_process:

# NEW
if cfg.use_lora:
    base_vla = AutoModelForVision2Seq.from_pretrained(...)
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla: PreTrainedModel = merged_vla.merge_and_unload()
    
    # Register custom dataset statistics in config
    merged_vla.config.norm_stats[cfg.dataset_name] = vla_dataset.dataset_statistics[cfg.dataset_name]
    
    if distributed_state.is_main_process:
```

**Purpose:** Automatically includes normalization statistics in the checkpoint config so inference can denormalize actions using `unnorm_key="my_robot_dataset"`.

---

#### Change B: CPU Training Modifications

**Note:** These changes are **only for CPU training**. For GPU training, use the GPU versions.

**B1. Device Initialization (Line ~95-105)**

```python
# OLD
assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
torch.cuda.empty_cache()

# NEW (CPU)
# assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
distributed_state = PartialState()
# torch.cuda.set_device(device_id := distributed_state.local_process_index)
# torch.cuda.empty_cache()
device_id = "cpu"
```

**B2. Remove DDP Wrapping (Line ~180)**

```python
# OLD
vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

# NEW (CPU)
# vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
```

**B3. Remove .module Accessors (Lines ~275, 310)**

When DDP is disabled on CPU, remove all `.module` accessors:

```python
# OLD (when using DDP on GPU)
vla.module.config.image_sizes
vla.module.vision_backbone.featurizer.patch_embed.num_patches
vla.module.save_pretrained(save_dir)

# NEW (CPU without DDP)
vla.config.image_sizes
vla.vision_backbone.featurizer.patch_embed.num_patches
vla.save_pretrained(save_dir)
```

**B4. Autocast for CPU (Line ~255)**

```python
# OLD (GPU)
with torch.autocast("cuda", dtype=torch.bfloat16):

# NEW (CPU)
with torch.autocast("cpu", dtype=torch.bfloat16):
```

**B5. Remove Distributed Barriers (Lines ~315, 340)**

Comment out both `dist.barrier()` calls for CPU training:

```python
# OLD
# Wait for processor and adapter weights to be saved by main process
dist.barrier()

# ... checkpoint save code ...

# Block on Main Process Checkpointing
dist.barrier()

#-------------------------------------------
# NEW (CPU)
# Wait for processor and adapter weights to be saved by main process
# dist.barrier()

# ... checkpoint save code ...

# Block on Main Process Checkpointing
# dist.barrier()
```

**Purpose:** CPU training doesn't support multi-GPU operations (DDP, distributed barriers, CUDA-specific autocasting). These changes make the training loop CPU-compatible.

---

## How These Changes Work Together

### Training Flow:
1. **mixtures.py** → enables `--dataset_name my_robot_dataset` flag
2. **configs.py** → defines data structure (7-dim joint state + gripper, 7-dim actions)
3. **transforms.py** → converts HDF5 data to standard trajectory format
4. **finetune.py** → saves dataset stats in config for later use

### Inference Flow:
1. Load model from checkpoint (has `norm_stats` in config)
2. Use `vla.predict_action(..., unnorm_key="my_robot_dataset", do_sample=False)`
3. Model automatically denormalizes actions using saved statistics

---

## Usage

### Training:
On CPU (inside the `openvla` repo directory)
```bash
python vla-scripts/finetune.py \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name my_robot_dataset \
    --run_root_dir ./runs \
    --batch_size 1 \
    --max_steps 100 \
    --use_lora true \
    --save_steps 10
```
On Colab Pro with NVIDIA A100 GPU (inside the `openvla` repo directory)
```bash
cd /content/openvla
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --data_root_dir /root/tensorflow_datasets \
    --dataset_name my_robot_dataset \
    --run_root_dir /content/openvla/runs \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --max_steps 20000 \
    --shuffle_buffer_size 1000 \
    --image_aug true \
    --use_lora true \
    --lora_rank 32 \
    --save_steps 2000
```

### Inference:
Run the [server](api_server.py) on either A100 GPU or CPU for model inference. Ensure correct model checkpoint is provided in `vla` instantiation.
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

vla = AutoModelForVision2Seq.from_pretrained(
    # use finetuned weights
    "./runs/openvla-7b+my_robot_dataset+...", 
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
    attn_implementation="eager"
).to(device)    # cuda or cpu
```
Run the [client](api_client.py) on the laptop connected to the robot for passing robot states (image for model and end effector pose for the robot) between the server and robot.

---

## Notes

- All changes are **additive** — no existing dataset code was modified
- The dataset transform function is minimal; customize as needed for your HDF5 structure
- Normalization statistics are automatically computed during training via `RLDSDataset`
- No separate patching script needed after these changes (**Note:** If denormalization of action tokens fails due to missing dataset key in the data_statistics config file, use [patch_checkpoint_with_stats.py](openvla_utils/patch_checkpoint_with_stats.py) as workaround)
