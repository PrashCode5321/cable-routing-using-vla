import threading, time
import numpy as np
import cv2
import h5py
from utils.zed_camera import ZedCamera
from pathlib import Path

def position_printer(
    arm,
    zed: ZedCamera,
    stop_event: threading.Event,
    raw_samples: list,
    hz: float = 15.0,
) -> None:
    period = 1.0 / hz
    while not stop_event.is_set():
        loop_start = time.time()
        timestamp = loop_start
        pose = None
        state = None
        frame = None
        code1 = -1
        code2 = -1

        try:
            code1, pose = arm.get_position(is_radian=True)
            code2, state = arm.get_servo_angle(is_radian=True)
            frame = zed.image
            # print(f"[Position] {pose} | [State] {state} | [Frame Shape] {frame.shape if frame is not None else None}")
            if code1 == 0 and code2 == 0 and frame is not None:
                pose_h = np.append(np.asarray(pose, dtype=np.float32), 1.0)
                raw_samples.append(
                    {
                        "timestamp": timestamp,
                        "pose": pose_h,
                        "state": state,
                        "frame": frame.copy(),
                    }
                )
            else:
                print(f"[Position] get_position failed with code={code1, code2, frame is not None}")
        except Exception as exc:
            print(f"[Position] error: {exc}")

        elapsed = time.time() - loop_start
        sleep_time = max(0.0, period - elapsed)
        stop_event.wait(sleep_time)


def save_to_hdf5(processed: dict, output_dir: str = "./demonstrations", 
                task_name: str = "clip_pickup", success: bool = True) -> str:
    """Save processed episode data to HDF5 format for OpenVLA training.
    
    Args:
        processed: Dict from post_process_samples() containing observations and actions
        output_dir: Directory to save episodes
        task_name: Language instruction/task description
        success: Whether episode was successful
        
    Returns:
        Path to saved HDF5 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next episode index
    existing = sorted(output_dir.glob("episode_*.hdf5"))
    episode_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    
    fname = output_dir / f"episode_{episode_idx:04d}.hdf5"
    T = len(processed["joint_states"])
    
    with h5py.File(fname, "w") as f:
        # Observations group
        obs = f.create_group("observations")
        obs.create_dataset("images", data=processed["frames_224"], 
                          compression="gzip", compression_opts=4)
        obs.create_dataset("images_full", data=processed["frames_full"],
                          compression="gzip", compression_opts=4)
        obs.create_dataset("joint_states", data=processed["joint_states"])
        obs.create_dataset("ee_poses", data=processed["ee_poses"])
        
        # Actions (gripper state included in ee_poses at index 6)
        f.create_dataset("joint_actions", data=processed["action_joint_states"])
        f.create_dataset("ee_actions", data=processed["action_ee_poses"])
        f.create_dataset("timestamps", data=processed["timestamps"])
        
        # Episode metadata
        meta = f.create_group("episode_metadata")
        meta.attrs["language_instruction"] = task_name
        meta.attrs["success"] = success
        meta.attrs["num_steps"] = T
        meta.attrs["timestamp"] = time.time()
        meta.attrs["collection_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # RLDS-compatible format
        is_first = np.zeros(T, dtype=bool)
        is_first[0] = True
        is_last = np.zeros(T, dtype=bool)
        is_last[-1] = True
        is_terminal = is_last.copy()
        rewards = np.zeros(T, dtype=np.float32)
        rewards[-1] = 1.0 if success else 0.0
        
        f.create_dataset("is_first", data=is_first)
        f.create_dataset("is_last", data=is_last)
        f.create_dataset("is_terminal", data=is_terminal)
        f.create_dataset("rewards", data=rewards)
    
    print(f"[Save] Episode saved to {fname}")
    print(f"[Save] Episode {episode_idx:04d} | Steps: {T} | Success: {success}")
    return str(fname)


def post_process_samples(raw_samples: list) -> dict:
    """Convert raw streamed samples into model-ready arrays after capture ends.
    
    - Stacks raw samples into arrays
    - Converts EE pose translation from mm to m (keeping rotation in radians)
    - Gripper state (appended 1) represents closed gripper throughout action execution
    - Computes action deltas for training
    """
    if not raw_samples:
        return {
            "timestamps": np.array([], dtype=np.float32),
            "joint_states": np.empty((0, 8), dtype=np.float32),
            "ee_poses": np.empty((0, 7), dtype=np.float32),
            "frames_full": np.empty((0, 1242, 2208, 4), dtype=np.uint8),
            "frames_224": np.empty((0, 224, 224, 3), dtype=np.uint8),
            "action_joint_states": np.empty((0, 8), dtype=np.float32),
            "action_ee_poses": np.empty((0, 7), dtype=np.float32),
        }

    timestamps = np.array([sample["timestamp"] for sample in raw_samples], dtype=np.float32)
    joint_states = np.array([sample["state"] for sample in raw_samples], dtype=np.float32)
    ee_poses = np.array([sample["pose"] for sample in raw_samples], dtype=np.float32)
    
    # Convert EE pose translation from mm to m (first 3 elements)
    # Keep rotation in radians (elements 3-6) and gripper state (element 6)
    ee_poses[:, :3] /= 1000.0  # Convert translation from mm to m
    
    frames_full = np.stack([sample["frame"] for sample in raw_samples], axis=0)
    frames_224 = np.stack(
        [cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR) for frame in frames_full],
        axis=0,
    )

    # Compute action deltas (differences between consecutive frames)
    action_joint_states = np.diff(joint_states, axis=0)
    action_ee_poses = np.diff(ee_poses, axis=0)

    return {
        "timestamps": timestamps,
        "joint_states": joint_states,
        "ee_poses": ee_poses,
        "frames_full": frames_full,
        "frames_224": frames_224,
        "action_joint_states": action_joint_states,
        "action_ee_poses": action_ee_poses,
    }


def video_writer(
    raw_samples: list,
    stop_event: threading.Event,
    output_dir: str = "./episodes",
    fps: float = 10.0,
    frame_idx: int = 0,
) -> str:
    """Write frames from raw_samples to a video file in a separate thread.
    
    Args:
        raw_samples: List of sample dicts (populated by position_printer)
        stop_event: Threading event to signal when to stop
        output_dir: Directory to save video
        fps: Frames per second for video
        frame_idx: Index to match with HDF5 episode (auto-increment if None)
    
    Returns:
        Path to saved video file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next video index
    existing = sorted(output_dir.glob("episode_*.mp4"))
    video_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    
    video_path = output_dir / f"episode_{video_idx:04d}.mp4"
    
    # Will be initialized when first frame arrives
    writer = None
    frame_width = None
    frame_height = None
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    last_frame_count = 0
    
    print(f"[VideoWriter] Starting video writer (target: {fps} fps, output: {video_path})")
    
    try:
        while not stop_event.is_set():
            # Check if new frames arrived
            current_frame_count = len(raw_samples)
            
            if current_frame_count > last_frame_count and current_frame_count > 0:
                # Get the latest frame
                latest_frame = raw_samples[-1]["frame"]
                
                # Initialize writer on first frame
                if writer is None:
                    h, w = latest_frame.shape[:2]
                    frame_height, frame_width = h, w
                    writer = cv2.VideoWriter(
                        str(video_path), 
                        codec, 
                        fps, 
                        (frame_width, frame_height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {video_path}")
                    print(f"[VideoWriter] Initialized: {frame_width}x{frame_height} @ {fps} fps")
                
                # Write the frame (convert BGRA to BGR if needed)
                if latest_frame.shape[2] == 4:
                    # BGRA -> BGR
                    frame_to_write = cv2.cvtColor(latest_frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame_to_write = latest_frame
                
                writer.write(frame_to_write.astype(np.uint8))
                last_frame_count = current_frame_count
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.01)
        
        # Final write of any remaining frames
        if writer is not None:
            for i in range(last_frame_count, len(raw_samples)):
                frame = raw_samples[i]["frame"]
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                writer.write(frame.astype(np.uint8))
        
        print(f"[VideoWriter] ✓ Wrote {len(raw_samples)} frames to {video_path}")
        
    except Exception as e:
        print(f"[VideoWriter] ✗ Error: {e}")
        return None
    
    finally:
        if writer is not None:
            writer.release()
    
    return str(video_path)

