# Cable Routing with OpenVLA

Fine-tuning [OpenVLA](https://github.com/openvla/openvla) for robotic cable placement tasks using a **Lite6 robot arm** and **ZED camera**.

## Project Goal

Train a Vision-Language Action (VLA) model to manipulate a robot arm to **place cables in Y-shaped brackets** using visual observations and natural language instructions.

## System Architecture

### Hardware
- **Robot Arm**: Lite6 (UFACTORY)
- **Camera**: ZED 2i (RGB-D stereo)
- **Gripper**: Lite6 integrated gripper

### Software Stack
- **Model**: OpenVLA (7B parameter vision-language action model)
- **Framework**: Hugging Face Transformers + PyTorch
- **Data Format**: HDF5 with RLDS (Reinforcement Learning Dataset Standard)

## Data Collection Format

The dataset is collected and stored in HDF5 format following the structure below:

![Data Collection Format](media/data_collection.png)

**Dataset Structure**:
- **Observations**: RGB frames (1242×2208×3), resized frames (224×224×3), joint states (8D), end-effector states (7D)
- **Actions**: Joint action deltas (8D), end-effector action deltas (7D), timestamps
- **Metadata**: Task success indicator, language prompt, number of steps
- **RLDS**: Reinforcement learning dataset standard fields (is_first, is_last, is_terminal, rewards)

## Project Structure

```
├── stream.py               # Data collection pipeline (ZED camera + robot recording)
├── api_server.py          # FastAPI server for model inference
├── api_client.py          # Client for sending requests to API
├── vla_test.py            # Manual inference testing
├── openvla_utils/         # OpenVLA utilities
│   ├── finetune.py        # Fine-tuning script
│   ├── transforms.py      # Data augmentation
│   └── configs.py         # Configuration management
├── utils/                 # Helper utilities
│   ├── detector.py        # AprilTag-based bracket detection
│   ├── zed_camera.py      # Camera interface
│   ├── planner.py         # Motion planning
│   ├── record.py          # Data recording & HDF5 saving
│   └── vis_utils.py       # Visualization tools
├── my_robot_dataset/      # Custom dataset builder for Hugging Face
├── demonstrations/        # Collected demonstrations (video + metadata)
├── episodes/              # Raw episode recordings
└── media/                 # Documentation media
    ├── data_collection.png    # HDF5 format diagram
    └── episode_0102.hdf5      # Sample episode data
```

## Workflow

### 1. Data Collection
```bash
python stream.py --tag-ids 8 --fps 5
```
- Records synchronized RGB frames, joint states, and end-effector states
- Saves to HDF5 format with RLDS standard structure
- Generates demonstration videos

### 2. Fine-tuning OpenVLA
- Load the collated episodes to Google Drive in zip format
- Do this on Google Colab Pro. Use the notebook [`here`](openvla_colab.ipynb)
- Upload the `mixtures.py`, `configs.py`, `transforms.py` and `finetune.py` scripts along with `my_robot_dataset` directory to Colab notebook
- Run the entire notebook.

### 3. Inference
**API Server**
Launch the API server on GPU instance on Cloud. Ensure it has a minimum of 25GB vRAM.  
Load the [shell script](utils/startup.sh) there and run it.
```bash
python api_server.py
```
Then use the [client](api_client.py) to stream robot state and receive VLA output.

## Key Components

### Stream (Data Collection)
- **ZedCamera**: Captures synchronized RGB + depth at configurable FPS
- **BracketDetector**: Detects Y-bracket and cable using AprilTags
- **ActionPlanner**: Generates cable placement trajectories
- **Record**: Saves episodes to HDF5 with frame synchronization
- **Stream**: To run live demonstrations with the robot and save them for fine-tuning.

### API Server
- FastAPI endpoint for real-time inference
- Receives PIL images + task instructions
- Returns predicted joint/EE action deltas
- Automatic action denormalization using dataset statistics

## References

- **OpenVLA**: [openvla/openvla on HF Hub](https://github.com/openvla/openvla)
- **Lite6 Docs**: [UFACTORY Lite6 Manual](https://github.com/xarm-developer/xarm-python-sdk)
- **ZED Camera**: [Stereolabs ZED Documentation](https://www.stereolabs.com/docs/)
