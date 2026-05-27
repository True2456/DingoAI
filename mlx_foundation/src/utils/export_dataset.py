import os
import json
import glob

def export_premium_dataset():
    print("=== MLX Agent Trajectory Dataset Exporter ===")
    jsonl_files = glob.glob("data/*_trajectories.jsonl")
    if not jsonl_files:
        print("No generated trajectories found in data/ directory yet.")
        print("Please wait for Step 1 of the current iteration to finish generating trajectories.")
        return

    all_exported = []
    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        print(f"Processing file: {filename}...")
        
        with open(filepath, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                
                # Unique ID for the sample
                sample_id = f"mlx_agent_traj_{filename.split('_')[1]}_{i+1:04d}"
                
                instruction = sample.get("instruction", "")
                output = sample.get("output", "")
                success = sample.get("sandbox_success", True)
                teacher = sample.get("teacher_model", "Ensemble Teacher")
                task_id = sample.get("task_id", "")
                failed_attempt_count = sample.get("failed_attempt_count", 0)
                
                # Check for high-fidelity turn structure
                turns = []
                if "turns" in sample:
                    turns = sample["turns"]
                else:
                    # Low-fidelity fallback: reconstruct from joined strings
                    thoughts = sample.get("thought", "").split(" | ")
                    actions = sample.get("actions", "").split(" | ")
                    observations = sample.get("observation", "").split(" | ")
                    
                    num_turns = max(len(thoughts), len(actions), len(observations))
                    for turn_idx in range(num_turns):
                        t_thought = thoughts[turn_idx] if turn_idx < len(thoughts) else ""
                        t_action_str = actions[turn_idx] if turn_idx < len(actions) else "none: "
                        
                        # Parse action type and input
                        action_type = "none"
                        action_input = ""
                        if ":" in t_action_str:
                            parts = t_action_str.split(":", 1)
                            action_type = parts[0].strip()
                            action_input = parts[1].strip()
                        else:
                            action_type = t_action_str.strip()
                            
                        t_obs = observations[turn_idx] if turn_idx < len(observations) else ""
                        
                        turns.append({
                            "turn": turn_idx + 1,
                            "thought": t_thought,
                            "action": {
                                "type": action_type,
                                "input": action_input
                            },
                            "observation": {
                                "stdout": t_obs if "error" not in t_obs.lower() else "",
                                "stderr": t_obs if "error" in t_obs.lower() else "",
                                "success": "error" not in t_obs.lower()
                            }
                        })
                
                # Build premium ShareGPT-compatible conversation trace
                sharegpt_conversations = []
                # System prompt
                sharegpt_conversations.append({
                    "from": "system",
                    "value": "You are a local sandboxed Python programming agent. Solve the user's task using tool calls."
                })
                # User request
                sharegpt_conversations.append({
                    "from": "human",
                    "value": instruction
                })
                
                # Conversational trace turns
                for t in turns:
                    # Agent's response containing thought and action call
                    action_call = f"Action ({t['action']['type']}): {t['action']['input']}" if t['action']['type'] != "none" else "Done."
                    agent_message = f"[THOUGHT]{t['thought']}[END]\n[ACTION]{action_call}[END]"
                    
                    sharegpt_conversations.append({
                        "from": "gpt",
                        "value": agent_message
                    })
                    
                    # Tool response
                    if t['action']['type'] != "none":
                        tool_obs = t['observation']['stdout'] or t['observation']['stderr']
                        sharegpt_conversations.append({
                            "from": "tool",
                            "value": f"[OBS]{tool_obs}[END]"
                        })
                
                # Final response
                sharegpt_conversations.append({
                    "from": "gpt",
                    "value": f"[OUTPUT]{output}[END]"
                })
                
                # Premium commercial format
                premium_entry = {
                    "id": sample_id,
                    "metadata": {
                        "task_id": task_id,
                        "task_description": instruction,
                        "verified_sandbox_success": success,
                        "teacher_model_origin": teacher,
                        "failed_attempt_count": failed_attempt_count,
                        "source_file": filename,
                        "format_version": "1.1-HighFidelity"
                    },
                    "conversations": sharegpt_conversations,
                    "turns_trace": turns,
                    "failed_attempts": sample.get("failed_attempts", [])
                }
                all_exported.append(premium_entry)
            except Exception as ex:
                print(f"Error parsing sample: {ex}")

    output_path = "data/mlx_commercial_agent_trajectories_v1.json"
    with open(output_path, "w") as f:
        json.dump(all_exported, f, indent=2)
        
    print(f"\nSuccessfully compiled and exported {len(all_exported)} premium trajectories!")
    print(f"Premium dataset saved to: {output_path}")

if __name__ == "__main__":
    export_premium_dataset()
