import os
import argparse
import h5py
import numpy as np
import time
from pathlib import Path
from xarm.wrapper import XArmAPI
from utils import presets


def load_hdf5_episode(filepath: str) -> dict:
    """Load episode data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        # print(len(f), print(list(f.keys())))
        # print(len(f["rewards"][:]))
        return {"delta_actions": f["ee_actions"][:], "timestamps": f["timestamps"][:]}


def ee_pose_to_command(pose: np.ndarray, speed: int = 80) -> dict:
    """Convert EE pose [x, y, z, roll, pitch, yaw, gripper] to robot command.
    
    Args:
        pose: End effector pose in shape (7,) with last element as gripper state
        speed: Motion speed
    
    Returns:
        Dict with robot command parameters
    """
    x_mm, y_mm, z_mm = pose[0], pose[1], pose[2]
    roll_rad, pitch_rad, yaw_rad = pose[3], pose[4], pose[5]
    # gripper = pose[6]
    
    # Convert radians to degrees
    roll_deg = np.degrees(roll_rad)
    pitch_deg = np.degrees(pitch_rad)
    yaw_deg = np.degrees(yaw_rad)
    
    return {
        "x": float(x_mm),
        "y": float(y_mm),
        "z": float(z_mm),
        "roll": float(roll_deg),
        "pitch": float(pitch_deg),
        "yaw": float(yaw_deg),
        "is_radian": False,
        "speed": speed,
        "wait": True,
    }


def replay_episode(filepath: str, speed: int = 80, dry_run: bool = False) -> None:
    """Replay a recorded episode from HDF5 file.
    
    Args:
        filepath: Path to HDF5 episode file
        speed: Robot motion speed (1-100)
        dry_run: If True, only print commands without executing
    """
    try:
        arm = XArmAPI(presets.ROBOT_IP)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, presets.GRIPPER_LENGTH, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)
        arm.move_gohome(speed=presets.SPEED, wait=True)
        arm.close_lite6_gripper(sync=True)

        # Load episode data
        print(f"Loading episode from {filepath}...")
        episode = load_hdf5_episode(filepath)
        
        print(f"Replaying {len(episode['delta_actions'])} steps...\n")
        
        for step_idx, frame in enumerate(episode["delta_actions"]):
            # Get current position: [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]
            _, pose_mm = arm.get_position(is_radian=True)
            
            # Convert current pose to meters (to match delta units)
            pose_m = np.array([
                pose_mm[0] / 1000.0,  # mm -> m
                pose_mm[1] / 1000.0,
                pose_mm[2] / 1000.0,
                pose_mm[3],           # radians (no conversion)
                pose_mm[4],
                pose_mm[5],
            ])
            
            # Add delta (delta is in meters for x,y,z and radians for angles)
            target_pose_m = pose_m + frame[:6]
            
            # Convert back to mm for robot command
            target_pose_mm = np.array([
                target_pose_m[0] * 1000.0,
                target_pose_m[1] * 1000.0,
                target_pose_m[2] * 1000.0,
                target_pose_m[3],
                target_pose_m[4],
                target_pose_m[5],
            ])
            
            # Create command
            cmd = ee_pose_to_command(target_pose_mm)
            
            # Gripper control
            print("gripper state:", frame[6])
            if frame[6] > 0.5:
                arm.open_lite6_gripper(sync=True)
            else:
                arm.close_lite6_gripper(sync=True)
            
            print(f"Step {step_idx+1}: delta={frame[:3]} -> target={target_pose_mm[:3]}")
            ret = arm.set_position(**cmd)
            if ret != 0:
                print(f"  Warning: returned {ret}")
            time.sleep(0.1)
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay recorded robot demonstrations from HDF5 files")
    parser.add_argument("--episode", type=str, help="Path to HDF5 episode file to replay")
    parser.add_argument("--demo-dir", type=str, default="./episodes", 
                       help="Demonstrations directory (if --episode not specified, plays latest)")
    parser.add_argument("--speed", type=int, default=80, 
                       help="Robot motion speed (1-100)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print trajectory without executing")
    parser.add_argument("--list", action="store_true",
                       help="List all available episodes")
    args = parser.parse_args()
    
    demo_path = Path(args.demo_dir)
    
    # List episodes if requested
    if args.list:
        episodes = sorted(demo_path.glob("episode_*.hdf5"))
        if not episodes:
            print(f"No episodes found in {args.demo_dir}")
        else:
            print(f"Available episodes in {args.demo_dir}:")
            for ep in episodes:
                print(f"  {ep.name}")
        exit(0)
    
    # Determine which episode to play
    if args.episode:
        filepath = args.episode
    else:
        # Find latest episode
        episodes = sorted(demo_path.glob("episode_*.hdf5"))
        if not episodes:
            print(f"No episodes found in {args.demo_dir}")
            exit(1)
        filepath = str(episodes[-1])
        print(f"No episode specified, using latest: {filepath}\n")
    
    # Replay
    replay_episode(filepath, speed=args.speed, dry_run=args.dry_run)
