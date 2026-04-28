"""
Langgraph agent for routing cable through bracket clips.

The agent:
1. Parses natural language routing instructions into bracket IDs
2. Detects all brackets in the workspace
3. Filters detected brackets by requested IDs
4. Builds combined motion plan for all brackets in sequence
5. Executes the combined plan and streams data
6. Handles errors when requested brackets are not found

Bracket ID mapping:
  - y-clip: multiples of 3 (3, 6, 9, 12, ...)
  - c-clip: multiples of 4 (4, 8, 12, ...)
  - r-clip: multiples of 5 (5, 10, 15, ...)
"""

import json
import re
import cv2
import threading
import time
from typing import Any, Literal, List, Tuple, Dict
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
# import anthropic
import numpy as np

from utils.detector import BracketDetector
from utils.planner import ActionPlanner
from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from utils.record import position_printer, post_process_samples


# ============================================================================
# State Definition
# ============================================================================

@dataclass
class RoutingState:
    """State for the routing agent."""
    user_instruction: str = ""                    # Original natural language input
    requested_bracket_ids: List[int] = field(default_factory=list)  # Parsed bracket IDs [8, 5, ...]
    detected_brackets: Dict[int, np.ndarray] = field(default_factory=dict)  # {tag_id: pose}
    filtered_bracket_ids: List[int] = field(default_factory=list)  # Requested IDs that were detected
    bracket_clip_types: Dict[int, str] = field(default_factory=dict)  # {tag_id: 'y'/'c'/'r'}
    combined_motion_plan: List[Dict] = field(default_factory=list)  # Merged plan for all brackets
    raw_samples: List[Dict] = field(default_factory=list)  # Streamed data during execution
    processed_data: Dict = None                    # Post-processed data
    execution_results: List[Dict] = field(default_factory=list)  # Results of execution
    error_message: str = None                     # Error message if something fails
    status: str = "parsing"                       # Current status
    zed_camera: ZedCamera = None                  # Shared camera instance
    bracket_detector: BracketDetector = None
    action_planner: ActionPlanner = None          # Shared planner instance
    motion_stream_seconds: float = 3.0            # Seconds to stream after motion


# ============================================================================
# Utility Functions
# ============================================================================

def determine_clip_type(bracket_id: int) -> str:
    """Determine clip type from bracket ID.
    
    - y-clip: multiples of 3
    - c-clip: multiples of 4
    - r-clip: multiples of 5
    
    If multiple match, prefer highest priority: r > c > y
    """
    clip_types = []
    
    if bracket_id % 3 == 0:
        clip_types.append('y')
    if bracket_id % 4 == 0:
        clip_types.append('c')
    if bracket_id % 7 == 0:
        clip_types.append('r')
    
    if not clip_types:
        return None
    
    # Priority: r-clip > c-clip > y-clip
    priority = {'r': 3, 'c': 2, 'y': 1}
    return max(clip_types, key=lambda ct: priority[ct])


def extract_bracket_ids_simple(instruction: str) -> list[int]:
    """Extract bracket IDs using simple regex patterns."""
    patterns = [
        r'(\d+)(?:\s+and\s+then\s+)?(\d+)',           # "8 and then 5" or "8 5"
        r'from\s+(\d+)\s+to\s+(\d+)',                  # "from 8 to 5"
        r'bracket[s]?\s+(\d+)(?:[,\s]+(\d+))*',        # "brackets 8 and 5"
        r'(?:id|tag)[s]?\s+(\d+)(?:[,\s]+(\d+))*',     # "id 8, 5"
        r'through\s+(\d+).*?(?:then|and)\s+(\d+)',     # "through 8 then 5"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, instruction, re.IGNORECASE)
        if matches:
            ids = []
            if isinstance(matches[0], tuple):
                for match in matches:
                    for group in match:
                        if group:
                            ids.append(int(group))
            else:
                for match in matches:
                    ids.append(int(match))
            
            if ids:
                return ids
    
    # Fallback: extract all numbers
    numbers = re.findall(r'\d+', instruction)
    return [int(n) for n in numbers] if numbers else []


def extract_bracket_ids_with_llm(instruction: str) -> list[int]:
    """Use Claude to parse complex natural language into bracket IDs."""
    try:
        client = anthropic.Anthropic()
        
        prompt = f"""Extract the sequence of bracket IDs from this routing instruction.
        Instruction: "{instruction}"
        Return ONLY a JSON object with a single "bracket_ids" field containing an array of integers.
        Example: {{"bracket_ids": [8, 5]}}
        If no bracket IDs can be extracted, return {{"bracket_ids": []}}
        """
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        try:
            data = json.loads(response_text)
            bracket_ids = data.get("bracket_ids", [])
            return bracket_ids if isinstance(bracket_ids, list) else []
        except json.JSONDecodeError:
            numbers = re.findall(r'\d+', response_text)
            return [int(n) for n in numbers] if numbers else []
    
    except Exception as e:
        print(f"[Parser] LLM extraction failed: {e}")
        return []


# ============================================================================
# Node 1: Parse Natural Language -> Bracket IDs
# ============================================================================

def parse_routing_instruction(state: RoutingState) -> RoutingState:
    """Parse natural language instruction into a sequence of requested bracket IDs."""
    instruction = state.user_instruction.strip()
    
    print(f"\n[Parser] Processing instruction: '{instruction}'")
    
    # Try simple pattern matching first (fast path)
    bracket_ids = extract_bracket_ids_simple(instruction)
    
    if bracket_ids:
        print(f"[Parser] Extracted bracket IDs: {bracket_ids}")
        state.requested_bracket_ids = bracket_ids
        state.status = "brackets_extracted"
        return state
    
    # # Fall back to LLM for complex natural language
    # print("[Parser] Attempting LLM-based extraction for complex instruction...")
    # bracket_ids = extract_bracket_ids_with_llm(instruction)
    
    # if bracket_ids:
    #     print(f"[Parser] LLM extracted bracket IDs: {bracket_ids}")
    #     state.requested_bracket_ids = bracket_ids
    #     state.status = "brackets_extracted"
    # else:
    #     state.error_message = f"Could not parse bracket IDs from: '{instruction}'"
    #     state.status = "parse_error"
    #     print(f"[Parser] Error: {state.error_message}")
    
    # return state


# ============================================================================
# Node 2: Detect all brackets and filter by requested IDs
# ============================================================================

def detect_and_filter_brackets(state: RoutingState) -> RoutingState:
    """Detect all brackets in workspace and filter by requested IDs."""
    if not state.requested_bracket_ids:
        state.error_message = "No bracket IDs requested"
        state.status = "error"
        return state
    
    print(f"\n[Detector] Initializing camera and detector...")
    
    try:
        state.zed_camera = ZedCamera()
        cv_image = state.zed_camera.image
        
        state.bracket_detector = BracketDetector(
            observation=cv_image,
            intrinsic=state.zed_camera.camera_intrinsic,
        )
        
        print(f"[Detector] Scanning workspace for all brackets...")
        all_detected = state.bracket_detector.identify_april_tag_ids()
        
        if not all_detected:
            state.error_message = "No brackets detected in workspace"
            state.status = "detection_error"
            print(f"[Detector] Error: {state.error_message}")
            state.zed_camera.close()
            return state
        
        # Store all detected brackets
        for tag_id, pose in all_detected:
            state.detected_brackets[tag_id] = pose
            state.bracket_clip_types[tag_id] = determine_clip_type(tag_id)
            print(f"[Detector] Bracket {tag_id} detected ({state.bracket_clip_types[tag_id].upper()}-clip)")
        
        # Filter by requested IDs (preserve order)
        state.filtered_bracket_ids = [
            bid for bid in state.requested_bracket_ids 
            if bid in state.detected_brackets
        ]
        
        # Check if all requested brackets were found
        missing = set(state.requested_bracket_ids) - set(state.filtered_bracket_ids)
        if missing:
            state.error_message = f"The following requested brackets were not found: {sorted(missing)}"
            state.status = "missing_brackets"
            print(f"[Detector] Warning: {state.error_message}")
            state.zed_camera.close()
            return state
        
        print(f"[Detector] All requested brackets detected: {state.filtered_bracket_ids}")

        for bracket in state.filtered_bracket_ids:
            pose = state.detected_brackets[bracket]
            draw_pose_axes(image=cv_image, camera_intrinsic=state.zed_camera.camera_intrinsic, pose=pose)
        
        cv2.namedWindow("Verifying Tag Poses", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Tag Poses", 1280, 720)
        cv2.imshow("Verifying Tag Poses", cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord("k"):
            raise("Aborted by user.")
        
        state.status = "ready_to_plan"
        return state
        
    except Exception as e:
        state.error_message = f"Detection error: {str(e)}"
        state.status = "detection_error"
        print(f"[Detector] Error: {state.error_message}")
        if state.zed_camera:
            state.zed_camera.close()
        return state


# ============================================================================
# Node 3: Build combined motion plan
# ============================================================================

def build_combined_plan(state: RoutingState) -> RoutingState:
    """Build a combined motion plan for all requested brackets in sequence."""
    if state.status != "ready_to_plan":
        return state
    
    if not state.filtered_bracket_ids:
        state.error_message = "No brackets to plan for"
        state.status = "error"
        return state
    
    print(f"\n[Planner] Building combined motion plan...")
    
    try:
        # Initialize planner with detector camera pose
        state.action_planner = ActionPlanner(camera_pose=state.bracket_detector.camera_pose)
        state.action_planner.arm.close_lite6_gripper(sync=True)
        
        # Build plan for each bracket
        combined_plan = []
        for bracket_id in state.filtered_bracket_ids:
            pose = state.detected_brackets[bracket_id]
            clip_type = state.bracket_clip_types[bracket_id]
            
            print(f"[Planner] Building {clip_type.upper()}-clip plan for bracket {bracket_id}...")
            
            if clip_type == 'y':
                bracket_plan = state.action_planner.y_clip_plan(clip_pose=pose)
            elif clip_type == 'c':
                bracket_plan = state.action_planner.c_clip_plan(clip_pose=pose)
            elif clip_type == 'r':
                bracket_plan = state.action_planner.r_clip_plan(clip_pose=pose)
            else:
                raise ValueError(f"Unknown clip type: {clip_type}")
            
            # Add plan steps to combined plan
            combined_plan.extend(bracket_plan)
            print(f"[Planner] Added {len(bracket_plan)} steps for bracket {bracket_id}")
        
        state.combined_motion_plan = combined_plan
        print(f"[Planner] Combined plan contains {len(combined_plan)} total steps")
        state.status = "ready_to_execute"
        return state
        
    except Exception as e:
        state.error_message = f"Planning error: {str(e)}"
        state.status = "planning_error"
        print(f"[Planner] Error: {state.error_message}")
        if state.action_planner:
            state.action_planner.shutdown()
        if state.zed_camera:
            state.zed_camera.close()
        return state


# ============================================================================
# Node 4: Execute combined plan with streaming
# ============================================================================

def execute_combined_plan(state: RoutingState) -> RoutingState:
    """Execute the combined motion plan and stream sensor data."""
    if state.status != "ready_to_execute":
        return state
    
    print(f"\n[Executor] Starting execution of combined plan...")
    
    try:
        # Start position monitor thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=position_printer,
            args=(state.action_planner.arm, state.zed_camera, stop_event, state.raw_samples, 10.0),
            daemon=True,
        )
        monitor_thread.start()
        
        # Execute the combined plan
        print(f"[Executor] Executing {len(state.combined_motion_plan)} motion steps...")
        state.action_planner.execute_plan(state.combined_motion_plan)
        
        print(f"[Executor] Motion execution complete, streaming additional {state.motion_stream_seconds}s...")
        stop_event.wait(state.motion_stream_seconds)
        
        # Stop streaming
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        
        print(f"[Executor] ✓ Successfully executed plan for brackets: {state.filtered_bracket_ids}")
        
        for bracket_id in state.filtered_bracket_ids:
            state.execution_results.append({
                "bracket_id": bracket_id,
                "clip_type": state.bracket_clip_types[bracket_id],
                "status": "success"
            })
        
        state.status = "post_processing"
        return state
        
    except Exception as e:
        state.error_message = f"Execution error: {str(e)}"
        state.status = "execution_error"
        print(f"[Executor] ✗ Error: {state.error_message}")
        
        for bracket_id in state.filtered_bracket_ids:
            state.execution_results.append({
                "bracket_id": bracket_id,
                "clip_type": state.bracket_clip_types[bracket_id],
                "status": "error",
                "error": state.error_message
            })
        
        return state
    
    finally:
        if state.action_planner:
            state.action_planner.shutdown()
        if state.zed_camera:
            state.zed_camera.close()


# ============================================================================
# Node 5: Post-process and save data
# ============================================================================

def process_and_save_data(state: RoutingState) -> RoutingState:
    """Post-process streamed data."""
    if state.status not in ["post_processing", "execution_error"]:
        return state
    
    if not state.raw_samples:
        print(f"[PostProcess] No samples to process")
        state.status = "complete"
        return state
    
    print(f"\n[PostProcess] Processing {len(state.raw_samples)} raw samples...")
    
    try:
        processed_data = post_process_samples(state.raw_samples)
        print(f"[PostProcess] Data shapes:")
        print(f"  - joint_states: {processed_data['joint_states'].shape}")
        print(f"  - ee_poses: {processed_data['ee_poses'].shape}")
        print(f"  - frames_224: {processed_data['frames_224'].shape}")
        print(f"  - action_joint_states: {processed_data['action_joint_states'].shape}")
        print(f"  - action_ee_poses: {processed_data['action_ee_poses'].shape}")
        
        state.status = "complete"
        return state
        
    except Exception as e:
        state.error_message = f"Post-processing error: {str(e)}"
        state.status = "postprocess_error"
        print(f"[PostProcess] Error: {state.error_message}")
        return state


# ============================================================================
# Routing Logic
# ============================================================================

ERROR_STATUSES = {"parse_error", "detection_error", "missing_brackets", 
                  "planning_error", "execution_error", "postprocess_error"}

def should_error(state: RoutingState) -> bool:
    """Check if current state indicates an error."""
    return state.status in ERROR_STATUSES


# ============================================================================
# Graph Construction
# ============================================================================

def build_routing_graph():
    """Build the Langgraph routing agent with sequential flow and error handling.
    
    Flow: parse → detect → plan → execute → process → END
    Error routing: any error status → error → END
    """
    workflow = StateGraph(RoutingState)
    
    # Add nodes
    workflow.add_node("parse", parse_routing_instruction)
    workflow.add_node("detect", detect_and_filter_brackets)
    workflow.add_node("plan", build_combined_plan)
    workflow.add_node("execute", execute_combined_plan)
    workflow.add_node("process", process_and_save_data)
    workflow.add_node("error", lambda state: state)  # Error handler (no-op)
    
    # Set entry point
    workflow.set_entry_point("parse")
    
    # Sequential conditional flow with error handling
    workflow.add_conditional_edges(
        "parse",
        lambda state: "error" if should_error(state) else "detect",
        {"error": "error", "detect": "detect"}
    )
    
    workflow.add_conditional_edges(
        "detect",
        lambda state: "error" if should_error(state) else "plan",
        {"error": "error", "plan": "plan"}
    )
    
    workflow.add_conditional_edges(
        "plan",
        lambda state: "error" if should_error(state) else "execute",
        {"error": "error", "execute": "execute"}
    )
    
    workflow.add_conditional_edges(
        "execute",
        lambda state: "error" if should_error(state) else "process",
        {"error": "error", "process": "process"}
    )
    
    # End of sequential flow
    workflow.add_edge("process", END)
    workflow.add_edge("error", END)
    
    return workflow.compile()


# ============================================================================
# Main Interface
# ============================================================================

def route_cable(instruction: str, stream_seconds: float = 3.0, task_name: str = "cable_routing") -> dict[str, Any]:
    """Route cable through brackets based on natural language instruction.
    
    Args:
        instruction: Natural language routing instruction, e.g., "route through 8 then 5"
        stream_seconds: Seconds to continue streaming after motion completes
        task_name: Task description for logging
    
    Returns:
        Dictionary with execution results and processed data
    """
    graph = build_routing_graph()
    graph.get_graph().draw_mermaid_png(output_file_path=str("flow.png"))
    initial_state = RoutingState(
        user_instruction=instruction,
        motion_stream_seconds=stream_seconds
    )
    
    print("=" * 80)
    print(f"🤖 Starting routing agent")
    print(f"   Instruction: '{instruction}'")
    print(f"   Task: {task_name}")
    print("=" * 80)
    
    result = graph.invoke(initial_state)
    
    print(f"📊 Routing Complete")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Route cable through brackets using natural language")
    parser.add_argument("instruction", type=str, help="Natural language routing instruction")
    parser.add_argument("--stream-seconds", type=float, default=1.0, help="Seconds to stream after motion")
    parser.add_argument("--task-name", type=str, default="cable_routing", help="Task description")
    args = parser.parse_args()
    
    result = route_cable(
        args.instruction, 
        stream_seconds=args.stream_seconds,
        task_name=args.task_name
    )
    result = {k: v for k, v in result.items()
              if k in [
                  "user_instruction", "requested_bracket_ids", 
                  "bracket_clip_types", "execution_results"
            ]}
    print(result)