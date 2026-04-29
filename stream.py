import os
import glob
import gc

# Prevent Qt font warnings from OpenCV HighGUI by pointing to system fonts.
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

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
from utils import logger
import logging

logger = logging.getLogger("VLA")

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
        logger.info("ZedCamera created")
        
        cv_image = zed.image
        detector = BracketDetector(
            observation=cv_image,
            intrinsic=zed.camera_intrinsic,
        )

        planner = ActionPlanner(camera_pose=detector.camera_pose)

        logger.info("Scanned the workspace")
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
            logger.warning("Aborted by user.")
            return None

        # Start position monitor thread (10 Hz)
        monitor_thread = threading.Thread(
            target=position_printer,
            args=(planner.arm, zed, stop_event, raw_samples, float(fps), planner),
            daemon=True,
        )
        monitor_thread.start()
        logger.info("Capturing raw synced data (poses, states, frames) in memory.")

        # Start video writer thread (writes frames to MP4 in parallel)
        video_thread = threading.Thread(
            target=video_writer,
            args=(raw_samples, stop_event, "./demonstrations", float(fps)),
            daemon=True,
        )
        video_thread.start()
        logger.info("Recording video to demonstrations/episode_XXXX.mp4")
        time.sleep(0.5)

        plan = []
        planner.arm.close_lite6_gripper(sync=True)
        planner.set_gripper_state(0.0)
        for result in results:
            tag_id, t_cam_clip = result
            if tag_id not in tag_ids:
                logger.info(f"Skipping tag ID {tag_id} as it's not in the specified list {tag_ids}")
                continue
        

            logger.info("Starting plan execution...")
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
        # 1. Stop threads FIRST (before closing resources)
        logger.info("[Cleanup] Stopping threads...")
        stop_event.set()
        
        # Give threads time to notice the stop event
        time.sleep(0.2)
        
        if monitor_thread is not None:
            try:
                monitor_thread.join(timeout=5.0)
                if monitor_thread.is_alive():
                    logger.warning("[Cleanup] Position monitor did not stop gracefully")
                else:
                    logger.info("[Cleanup] Position monitor stopped")
            except Exception as e:
                logger.error(f"[Cleanup] Monitor join error: {e}")
        
        if video_thread is not None:
            try:
                video_thread.join(timeout=5.0)
                if video_thread.is_alive():
                    logger.warning("[Cleanup] Video writer did not stop gracefully")
                else:
                    logger.info("[Cleanup] Video writer stopped")
            except Exception as e:
                logger.error(f"[Cleanup] Video thread join error: {e}")
        
        # Critical: Wait longer to ensure threads fully exit and release C++ resources
        time.sleep(1.0)

        # 2. Close camera BEFORE shutting down planner
        if zed is not None:
            try:
                logger.info("[Cleanup] Closing ZedCamera...")
                zed.close()
                logger.info("[Cleanup] ZedCamera closed")
            except Exception as e:
                logger.error(f"[Cleanup] ZedCamera close error: {e}")
            finally:
                zed = None

        time.sleep(0.2)

        # 3. Shutdown planner (stop motion)
        if planner is not None:
            try:
                logger.info("[Cleanup] Shutting down planner...")
                planner.shutdown()
                logger.info("[Cleanup] Planner shutdown")
            except Exception as e:
                logger.error(f"[Cleanup] Planner shutdown error: {e}")
            finally:
                planner = None

        # 4. Wait for remaining stream time
        if plan_completed and stream_until is not None:
            remaining = stream_until - time.time()
            if remaining > 0:
                logger.info(f"[Cleanup] Continuing stream for {remaining:.1f}s after plan completion...")
                time.sleep(max(0, remaining))

        # 5. Process data
        try:
            logger.info("[Cleanup] Post-processing samples...")
            processed = post_process_samples(raw_samples, task_name=task_name)

            if processed["joint_states"].size > 0:
                logger.info(f"[Post] First joint state: {np.round(processed['joint_states'][0], 2)}")
                logger.info(f"[Post] First EE pose (translation in m, rotation in rad, gripper): {np.round(processed['ee_poses'][0], 4)}")
        except Exception as e:
            logger.error(f"[Cleanup] Post-processing error: {e}")

        # Ask for user approval and save/delete accordingly
        if processed is not None and processed["joint_states"].size > 0:
            approve = input("Do you approve of this episode? (y/n): ").lower() == 'y'
            
            if approve:
                logger.info("saving episode")
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
                        logger.warning(f"Deleted video: {latest_video}")
                    except Exception as e:
                        logger.error(f"Failed to delete video: {e}")
                logger.info("Episode discarded (HDF5 not saved)")
    
        # 6. Clean up raw samples and detector
        try:
            raw_samples.clear()
            detector = None
        except Exception as e:
            logger.error(f"[Cleanup] ✗ Cleanup error: {e}")

        # 7. Close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        logger.info("[Cleanup] ✓ Cleanup complete")
        
        # Final garbage collection
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
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for video recording and data capture")
    args = parser.parse_args()

    start = time.time()
    processed = run(tag_ids=args.tag_ids, task_name=args.task_name, fps=args.fps)
    logger.info("Episode streaming complete.")
    
    # Force garbage collection to clean up C++ objects before exit
    gc.collect()
    
    logger.info(f"Total time: {round(time.time() - start, 2)} s")
