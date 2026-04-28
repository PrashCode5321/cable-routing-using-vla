import os
import glob
import gc

# Prevent Qt font warnings from OpenCV HighGUI by pointing to system fonts.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import argparse
import threading
import time
import cv2
import numpy as np
from utils.detector import BracketDetector
from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from utils.planner import ActionPlanner
from utils.record import position_printer, save_to_hdf5, post_process_samples, video_writer
from typing import List


def run(tag_ids: List[int] = [8], post_plan_stream_seconds: float = 1.0, task_name: str = "clip_pickup", fps: int = 5) -> dict:
    zed = None
    detector = None
    planner = None
    stop_event = threading.Event()
    monitor_thread = None
    video_thread = None
    raw_samples = []
    processed = None
    plan_completed = False
    stream_until = None

    try:
        zed = ZedCamera()
        print("[Init] ✓ ZedCamera created")
        
        cv_image = zed.image
        detector = BracketDetector(
            observation=cv_image,
            intrinsic=zed.camera_intrinsic,
        )

        planner = ActionPlanner(camera_pose=detector.camera_pose)

        print("Scanned the workspace")
        results = detector.identify_april_tag_ids()
        if not results:
            raise RuntimeError(f"No objects found")

        for result in results:
            _, t_cam_clip = result
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam_clip)
        
        cv2.namedWindow("Verifying Tag Poses", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Tag Poses", 1280, 720)
        cv2.imshow("Verifying Tag Poses", cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord("k"):
            print("Aborted by user.")
            return None

        # Start position monitor thread (10 Hz)
        monitor_thread = threading.Thread(
            target=position_printer,
            args=(planner.arm, zed, stop_event, raw_samples, float(fps)),
            daemon=True,
        )
        monitor_thread.start()
        print("Capturing raw synced data (poses, states, frames) in memory.")

        # Start video writer thread (writes frames to MP4 in parallel)
        video_thread = threading.Thread(
            target=video_writer,
            args=(raw_samples, stop_event, "./demonstrations", float(fps)),
            daemon=True,
        )
        video_thread.start()
        print("Recording video to demonstrations/episode_XXXX.mp4")
        time.sleep(0.5)

        plan = []
        planner.arm.close_lite6_gripper(sync=True)
        for result in results:
            tag_id, t_cam_clip = result
            if tag_id not in tag_ids:
                print(f"Skipping tag ID {tag_id} as it's not in the specified list {tag_ids}")
                continue
        

            print("Starting plan execution...")
            if tag_id % 3 == 0:
                plan.extend(planner.y_clip_plan(clip_pose=t_cam_clip))
            elif tag_id % 4 == 0:
                plan.extend(planner.c_clip_plan(clip_pose=t_cam_clip))
            elif tag_id % 7 == 0:
                plan.extend(planner.r_clip_plan(clip_pose=t_cam_clip))
            else:
                raise ValueError("tag_id does not correspond to a known plan")
        
        planner.execute_plan(plan)

        plan_completed = True
        stream_until = time.time() + max(0.0, post_plan_stream_seconds)

    finally:
        # 1. Shutdown planner (stop motion)
        if planner is not None:
            try:
                print("[Cleanup] Shutting down planner...")
                planner.shutdown()
                print("[Cleanup] ✓ Planner shutdown")
            except Exception as e:
                print(f"[Cleanup] ✗ Planner shutdown error: {e}")

        # 2. Wait for remaining stream time
        if plan_completed and stream_until is not None:
            remaining = stream_until - time.time()
            if remaining > 0:
                print(f"[Cleanup] Continuing stream for {remaining:.1f}s after plan completion...")
                stop_event.wait(remaining)

        # 3. Stop threads and wait for them to finish
        print("[Cleanup] Stopping threads...")
        stop_event.set()
        
        if monitor_thread is not None:
            try:
                monitor_thread.join(timeout=2.0)
                print("[Cleanup] ✓ Position monitor stopped")
            except Exception as e:
                print(f"[Cleanup] ✗ Monitor join error: {e}")
        
        if video_thread is not None:
            try:
                video_thread.join(timeout=3.0)
                print("[Cleanup] ✓ Video writer stopped")
            except Exception as e:
                print(f"[Cleanup] ✗ Video thread join error: {e}")
        
        # 4. Process data (before closing camera, in case it's still needed)
        try:
            print("[Cleanup] Post-processing samples...")
            processed = post_process_samples(raw_samples, task_name=task_name)
            # print(f"[Post] Raw samples: {len(raw_samples)}")
            # print(f"[Post] joint_states shape: {processed['joint_states'].shape}")
            # print(f"[Post] ee_poses shape: {processed['ee_poses'].shape}")
            # print(f"[Post] frames_full shape: {processed['frames_full'].shape}")
            # print(f"[Post] frames_224 shape: {processed['frames_224'].shape}")
            # print(f"[Post] action_joint_states shape: {processed['action_joint_states'].shape}")
            # print(f"[Post] action_ee_poses shape: {processed['action_ee_poses'].shape}")

            if processed["joint_states"].size > 0:
                print("[Post] First joint state:", np.round(processed["joint_states"][0], 2))
                print("[Post] First EE pose (translation in m, rotation in rad, gripper):", np.round(processed["ee_poses"][0], 4))
        except Exception as e:
            print(f"[Cleanup] ✗ Post-processing error: {e}")

        # 5. Clean up detector before camera (avoid dangling pointers)
        if detector is not None:
            try:
                detector = None
            except Exception as e:
                print(f"[Cleanup] ✗ Detector cleanup error: {e}")

        # 6. Close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 7. Close camera (last step, after all threads are stopped)
        if zed is not None:
            try:
                print("[Cleanup] Closing ZedCamera...")
                zed.close()
                print("[Cleanup] ✓ ZedCamera closed")
            except Exception as e:
                print(f"[Cleanup] ✗ ZedCamera close error: {e}")
        
        print("[Cleanup] ✓ Cleanup complete")
        
        # Force garbage collection before returning
        gc.collect()
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-ids", type=int, nargs='+', default=[6],
                       help="Tag IDs to route through (space-separated, e.g., 8 5 3)")
    parser.add_argument("--task-name", type=str, default="clip_pickup", 
                       help="Language instruction for the task")
    parser.add_argument("--demo-dir", type=str, default="./episodes",
                       help="Directory to save demonstration episodes (HDF5 files)")
    args = parser.parse_args()

    start = time.time()
    processed = run(tag_ids=args.tag_ids, task_name=args.task_name)
    print("Episode streaming complete.")
    # Ask for user approval and save/delete accordingly
    if processed is not None and processed["joint_states"].size > 0:
        approve = input("Do you approve of this episode? (y/n): ").lower() == 'y'
        
        if approve:
            print("saving episode")
            save_to_hdf5(
                processed=processed,
                output_dir=args.demo_dir,
                task_name=args.task_name,
                success=True
            )
        else:
            # Delete the latest video file
            video_files = glob.glob(os.path.join("demonstrations", "*.mp4"))
            if video_files:
                latest_video = max(video_files, key=os.path.getctime)
                try:
                    os.remove(latest_video)
                    print(f"Deleted video: {latest_video}")
                except Exception as e:
                    print(f"Failed to delete video: {e}")
            print("Episode discarded (HDF5 not saved)")
    
    # Force garbage collection to clean up C++ objects before exit
    gc.collect()
    
    print("Total time:", round(time.time() - start, 2), "s")
