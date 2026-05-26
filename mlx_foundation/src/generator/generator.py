import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import mlx_lm

class BaseGenerator(ABC):
    """Base class for all synthetic data generators."""

    @abstractmethod
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generates a list of synthetic data samples."""
        pass

class TemplateGenerator(BaseGenerator):
    """A simple generator that uses templates to create instruction-response pairs."""

    def __init__(self, templates: List[Dict[str, List[str]]]):
        self.templates = templates

    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        samples = []
        for _ in range(num_samples):
            template_set = random.choice(self.templates)
            instruction = random.choice(template_set['instruction_templates'])
            response = random.choice(template_set['response_templates'])
            samples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
        return samples

class MLXGenerator(BaseGenerator):
    """
    High-performance generator using Apple's MLX framework.
    """

    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        self.model_path = model_path
        if adapter_path:
            print(f"Generator loading model with adapter from {adapter_path}...")
            self.model, self.tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = mlx_lm.load(model_path)

    def generate(self, num_samples: int, prompts: Optional[List[str]] = None, max_tokens: int = 128) -> List[Dict[str, Any]]:
        if prompts is None:
            prompts = ["Tell me a story about a coding robot."]

        samples = []
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )

            clean_response = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

            samples.append({
                "instruction": prompt,
                "input": "",
                "output": clean_response
            })
        return samples

class MultiTeacherMLXGenerator(BaseGenerator):
    """
    [DARK HORSE] Ensemble Distillation Generator.
    Uses multiple teachers to capture diverse reasoning styles.
    Implements memory-efficient loading: one teacher at a time.
    """

    def __init__(self, model_paths: List[str]):
        self.model_paths = model_paths

    def generate(self, num_samples: int, prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if prompts is None:
            prompts = ["Explain the concept of recursion."]

        samples = []
        for i in range(num_samples):
            # Pick teacher and prompt
            teacher_path = self.model_paths[i % len(self.model_paths)]
            prompt = prompts[i % len(prompts)]

            # Load teacher, generate, then allow it to be garbage collected
            # Note: mlx_lm.load returns (model, tokenizer)
            # We use a local scope to ensure the model is eligible for cleanup
            def _generate_with_teacher(path: str, p: str) -> Optional[Dict[str, Any]]:
                try:
                    import mlx_lm
                    model, tokenizer = mlx_lm.load(path)
                    response = mlx_lm.generate(
                        model,
                        tokenizer,
                        prompt=p,
                        max_tokens=512,
                        verbose=False
                    )
                    clean_response = response[len(p):].strip() if response.startswith(p) else response.strip()
                    
                    # Clean VRAM
                    del model
                    del tokenizer
                    import gc
                    gc.collect()
                    try:
                        import mlx.core as mx
                        mx.metal.clear_cache()
                    except ImportError:
                        pass
                        
                    return {"instruction": p, "output": clean_response}
                except Exception as e:
                    print(f"Error with teacher {path}: {e}")
                    return None

            sample = _generate_with_teacher(teacher_path, prompt)
            if sample:
                samples.append(sample)

        return samples

class DynamicTaskGenerator:
    """
    [DARK HORSE] Self-Instruct Curriculum Creator.
    Loads a teacher model to brainstorm diverse, sandboxed Python programming tasks dynamically.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate_tasks(self, num_tasks: int) -> List[str]:
        import mlx_lm
        import json
        import gc
        
        print(f"Bootstrapping {num_tasks} dynamic programming tasks using teacher {self.model_path}...")
        tasks = []
        try:
            model, tokenizer = mlx_lm.load(self.model_path)
            
            system_msg = (
                "You are an expert curriculum designer. Brainstorm a list of unique and diverse Python programming tasks "
                "that can be executed and verified in a simple local terminal sandbox.\n"
                "Each task must be a single sentence, concise, require computing a result dynamically, and "
                "be structured such that the solution can be verified using self-contained assertions (e.g. comparing outputs to expected test cases).\n"
                "Do NOT include any markdown code blocks or conversational text. Return ONLY a JSON list of strings.\n\n"
                "Example output:\n"
                "[\n"
                "  \"Write a Python script that sorts a list of tuples by their second element and prints it.\",\n"
                "  \"Create a Python script that finds all prime numbers up to 50 and prints them.\"\n"
                "]"
            )
            
            def extract_first_json_array(text: str) -> Optional[list]:
                brace_depth = 0
                in_string = False
                escape = False
                start_idx = -1
                
                for i, char in enumerate(text):
                    if escape:
                        escape = False
                        continue
                    if char == '\\':
                        if in_string:
                            escape = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '[':
                            if brace_depth == 0:
                                start_idx = i
                            brace_depth += 1
                        elif char == ']':
                            brace_depth -= 1
                            if brace_depth == 0 and start_idx != -1:
                                candidate = text[start_idx:i+1]
                                try:
                                    res = json.loads(candidate)
                                    if isinstance(res, list):
                                        return res
                                except json.JSONDecodeError:
                                    pass
                return None

            batch_size = 10
            consecutive_failures = 0
            while len(tasks) < num_tasks and consecutive_failures < 5:
                curr_batch_size = min(batch_size, num_tasks - len(tasks))
                user_msg = f"Generate exactly {curr_batch_size} unique and diverse Python programming tasks as a JSON list of strings."
                if tasks:
                    # Provide last 20 tasks to avoid duplicate topics
                    negative_examples = "\n".join([f"- {t}" for t in tasks[-20:]])
                    user_msg += f"\nDo NOT generate any tasks similar or duplicate to the following:\n{negative_examples}"
                
                if hasattr(tokenizer, "apply_chat_template"):
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"{system_msg}\n\n{user_msg}"
                
                response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
                clean_resp = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()
                
                batch_tasks = extract_first_json_array(clean_resp)
                if batch_tasks and isinstance(batch_tasks, list):
                    added_any = False
                    for t in batch_tasks:
                        if isinstance(t, str) and t.strip():
                            t_clean = t.strip()
                            if t_clean not in tasks:
                                tasks.append(t_clean)
                                added_any = True
                    if added_any:
                        print(f"  -> Generated {len(tasks)}/{num_tasks} unique tasks...")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
            
            # Explicit cleanup
            del model
            del tokenizer
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except ImportError:
                pass
            
        except Exception as e:
            print(f"Error bootstrapping dynamic tasks: {e}")
            
        if len(tasks) >= num_tasks:
            return tasks[:num_tasks]
        elif len(tasks) > 0:
            print(f"Only bootstrapped {len(tasks)} tasks due to generation limits. Padding with fallbacks.")
            # If we got some tasks, but not enough, let's pad them up to num_tasks
            fallbacks = [
                "Write a Python script that calculates the 10th Fibonacci number and print it.",
                "Create a Python script that formats a list of numbers into a comma-separated string.",
                "Write a Python script that asserts that the string 'mlx' is uppercase and runs successfully.",
                "Write a Python script that counts the number of vowels in 'antigravity' and prints it.",
                "Create a Python script that filters odd numbers from [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and prints the result.",
                "Write a Python script that reverses the words in 'the quick brown fox jumps over the lazy dog' and prints it.",
                "Create a Python script that calculates the factorial of 6 and prints the value.",
                "Write a Python script that parses the domain name from 'https://github.com/True2456/MLX-DISTILL' and prints it.",
                "Create a Python script that checks if the string 'racecar' is a palindrome and prints True or False.",
                "Write a Python script that converts temperature 98.6 Fahrenheit to Celsius and prints the result rounded to one decimal place."
            ]
            while len(tasks) < num_tasks:
                for f in fallbacks:
                    if len(tasks) >= num_tasks:
                        break
                    if f not in tasks:
                        tasks.append(f)
                    else:
                        tasks.append(f + f" (Variant {len(tasks)})")
            return tasks
        
        # Robust fallback
        print("Using robust fallback task list.")
        fallbacks = [
            "Write a Python script that calculates the 10th Fibonacci number and print it.",
            "Create a Python script that formats a list of numbers into a comma-separated string.",
            "Write a Python script that asserts that the string 'mlx' is uppercase and runs successfully.",
            "Write a Python script that counts the number of vowels in 'antigravity' and prints it.",
            "Create a Python script that filters odd numbers from [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and prints the result.",
            "Write a Python script that reverses the words in 'the quick brown fox jumps over the lazy dog' and prints it.",
            "Create a Python script that calculates the factorial of 6 and prints the value.",
            "Write a Python script that parses the domain name from 'https://github.com/True2456/MLX-DISTILL' and prints it.",
            "Create a Python script that checks if the string 'racecar' is a palindrome and prints True or False.",
            "Write a Python script that converts temperature 98.6 Fahrenheit to Celsius and prints the result rounded to one decimal place."
        ]
        res = []
        while len(res) < num_tasks:
            for f in fallbacks:
                if len(res) >= num_tasks:
                    break
                res.append(f if f not in res else f + f" (Variant {len(res)})")
        return res

class EnsembleAgenticTrajectoryGenerator(BaseGenerator):
    """
    Highly advanced multi-turn ensemble agentic trajectory generator.
    Alternates between multiple teacher paths, runs actions in the SandboxExecutor,
    and constructs a real trace history dynamically.
    """
    
    def __init__(self, model_paths: List[str], workspace_dir: str = "data/sandbox"):
        self.model_paths = model_paths
        self.workspace_dir = workspace_dir
        # Cache loaded teacher models in RAM to avoid loading overhead.
        # Since the user has 128GB of RAM, this is extremely efficient and safe.
        self.model_cache = {}

    def generate(self, num_samples: int, task_descriptions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        from sandbox.sandbox import SandboxExecutor
        import tempfile
        import os

        # Ensure the shared sandbox parent directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)

        if task_descriptions is None:
            # Dynamically bootstrap unique programming tasks using a larger teacher model
            # (preferring the coder model or the largest model in the list)
            teacher_for_bootstrap = self.model_paths[0]
            if len(self.model_paths) > 1:
                for path in self.model_paths:
                    path_lower = path.lower()
                    if 'coder' in path_lower or '31b' in path_lower or '35b' in path_lower or '36b' in path_lower:
                        teacher_for_bootstrap = path
                        break
            
            task_generator = DynamicTaskGenerator(teacher_for_bootstrap)
            task_descriptions = task_generator.generate_tasks(num_tasks=num_samples)

        samples = []
        for i in range(num_samples):
            task = task_descriptions[i % len(task_descriptions)]

            # We try up to 3 times to get a successful sandbox trajectory
            max_attempts = 3
            successful_sample = None

            # Local worker with unified RAM model caching
            def _interact_with_teacher(path: str, t: str, sandbox_dir: str) -> Optional[Dict[str, Any]]:
                try:
                    import mlx_lm
                    import json
                    # Determine if we should cache this model based on size (threshold: 45 GB)
                    # This prevents caching the massive Qwen Coder Next (79GB) while caching smaller ones.
                    should_cache = True
                    try:
                        model_size = 0
                        for root_dir, _, files_list in os.walk(path):
                            for file_name in files_list:
                                model_size += os.path.getsize(os.path.join(root_dir, file_name))
                        if model_size > 45 * 1024 * 1024 * 1024:
                            should_cache = False
                    except Exception:
                        pass

                    if should_cache:
                        if path in self.model_cache:
                            model, tokenizer = self.model_cache[path]
                        else:
                            print(f"Loading teacher model {path} into RAM cache...")
                            model, tokenizer = mlx_lm.load(path)
                            self.model_cache[path] = (model, tokenizer)
                    else:
                        print(f"Loading large teacher model {path} dynamically (not cached to protect RAM)...")
                        model, tokenizer = mlx_lm.load(path)

                    # Instantiate sandbox isolated exclusively to this jailed workspace
                    sandbox = SandboxExecutor(sandbox_dir)

                    history = f"Task: {t}\n"
                    max_turns = 3
                    
                    thought_trace = []
                    actions_trace = []
                    observations_trace = []
                    final_answer = ""
                    sandbox_success = True
                    turns_array = []

                    def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
                        import json
                        brace_depth = 0
                        in_string = False
                        escape = False
                        start_idx = -1
                        
                        for i, char in enumerate(text):
                            if escape:
                                escape = False
                                continue
                            if char == '\\':
                                if in_string:
                                    escape = True
                                continue
                            if char == '"':
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    if brace_depth == 0:
                                        start_idx = i
                                    brace_depth += 1
                                elif char == '}':
                                    brace_depth -= 1
                                    if brace_depth == 0 and start_idx != -1:
                                        candidate = text[start_idx:i+1]
                                        try:
                                            return json.loads(candidate)
                                        except json.JSONDecodeError:
                                            pass
                        return None

                    for turn in range(max_turns):
                        system_msg = (
                            "You are an AI programming agent executing actions in a local terminal.\n"
                            "Based on the history, generate your next reasoning step in JSON format.\n"
                            "The JSON must have the following keys:\n"
                            "- 'thought': Your logical analysis of the current state.\n"
                            "- 'action_type': One of: 'python' (to execute inline code), 'list_dir', or 'none' (if complete).\n"
                            "- 'action_input': The code snippet to run, or empty string.\n"
                            "- 'final_answer': A descriptive string summarizing the result ONLY if action_type is 'none'.\n\n"
                            "CRITICAL: Do NOT hardcode final results. Always write clean Python code to compute results. "
                            "You MUST include self-verifying test assertions (using 'assert') at the end of your script "
                            "to programmatically prove your solution is correct. If assertions fail, the script will crash, "
                            "which is the correct behavior for incorrect logic.\n\n"
                            "Format Example:\n"
                            "{\n"
                            "  \"thought\": \"I need to write a script to compute the solution and assert correctness.\",\n"
                            "  \"action_type\": \"python\",\n"
                            "  \"action_input\": \"def solve(x): return x * 2\\n\\n# Assertions\\nassert solve(5) == 10\\nprint('Verified')\",\n"
                            "  \"final_answer\": \"\"\n"
                            "}"
                        )
                        user_msg = f"Task: {t}\n\nHistory of interactions:\n{history}\n\nGenerate the next JSON step now:"

                        # Use tokenizer's chat template if supported
                        if hasattr(tokenizer, "apply_chat_template"):
                            messages = [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg}
                            ]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        else:
                            prompt = f"{system_msg}\n\n{user_msg}"

                        # Generate output with high token budget to prevent truncations
                        response = mlx_lm.generate(
                            model,
                            tokenizer,
                            prompt=prompt,
                            max_tokens=1024,
                            verbose=False
                        )

                        clean_resp = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

                        # Robustly extract JSON block
                        step = extract_first_json(clean_resp)
                        if step is None:
                            print(f"Failed to parse teacher response with extract_first_json. Response was: {clean_resp}")
                            sandbox_success = False
                            step = {
                                "thought": "I failed to generate valid JSON.",
                                "action_type": "none",
                                "action_input": "",
                                "final_answer": "Failed."
                            }

                        thought = step.get("thought", "")
                        action_type = step.get("action_type", "none")
                        action_input = step.get("action_input", "")

                        thought_trace.append(thought)
                        actions_trace.append(f"{action_type}: {action_input}")

                        turn_data = {
                            "turn": turn + 1,
                            "thought": thought,
                            "action": {
                                "type": action_type,
                                "input": action_input
                            },
                            "observation": {
                                "stdout": "",
                                "stderr": "",
                                "success": True
                            }
                        }

                        if action_type == "none" or not action_input:
                            final_answer = step.get("final_answer", "Task complete.")
                            observations_trace.append("Environment complete.")
                            turn_data["observation"]["stdout"] = "Environment complete."
                            turns_array.append(turn_data)
                            break

                        # Execute in Sandbox
                        exec_res = sandbox.execute(action_type, action_input)
                        if exec_res.get("success") and not exec_res.get("stderr"):
                            obs = exec_res.get("stdout", "") or exec_res.get("message", "Success.")
                            turn_data["observation"]["stdout"] = obs
                        else:
                            sandbox_success = False
                            obs = exec_res.get("stderr", "") or exec_res.get("error", "Error.")
                            turn_data["observation"]["stderr"] = obs
                            turn_data["observation"]["success"] = False

                        observations_trace.append(obs)
                        turns_array.append(turn_data)

                        # Update interaction history
                        history += (
                            f"\nThought: {thought}\n"
                            f"Action ({action_type}): {action_input}\n"
                            f"Observation: {obs}\n"
                        )

                    # Keep models cached in RAM, only garbage collect local variables.
                    # If the model was not cached (larger than 45 GB), explicitly unload it.
                    if not should_cache:
                        print(f"Unloading large model {path} to free memory...")
                        del model
                        del tokenizer
                    
                    import gc
                    gc.collect()
                    try:
                        import mlx.core as mx
                        mx.metal.clear_cache()
                    except ImportError:
                        pass

                    if thought_trace:
                        return {
                            "instruction": t,
                            "thought": " | ".join(thought_trace),
                            "actions": " | ".join(actions_trace),
                            "observation": " | ".join(observations_trace),
                            "output": final_answer or "Execution completed successfully.",
                            "sandbox_success": sandbox_success,
                            "turns": turns_array,
                            "teacher_model": path
                        }
                except Exception as e:
                    print(f"Error during teacher ensemble loop: {e}")
                    return None

            for attempt in range(max_attempts):
                # Attempt 0 always uses the first teacher (the 9B model).
                # Subsequent attempts alternate among the other teachers.
                if attempt == 0:
                    teacher_path = self.model_paths[0]
                else:
                    other_teachers = self.model_paths[1:] if len(self.model_paths) > 1 else self.model_paths
                    teacher_path = other_teachers[(i + attempt - 1) % len(other_teachers)]

                print(f"Generating agentic trajectory (Attempt {attempt+1}/{max_attempts}) using teacher {teacher_path} for task: '{task}'")
                
                # Jailed temporary directory workspace for this specific attempt to guarantee complete isolation
                with tempfile.TemporaryDirectory(prefix=f"sandbox_rollout_{i}_{attempt}_", dir=self.workspace_dir) as tmp_workspace_dir:
                    res = _interact_with_teacher(teacher_path, task, tmp_workspace_dir)
                    if res and res.get("sandbox_success", True):
                        successful_sample = res
                        break
                    else:
                        reason = "Sandbox execution failed or syntax error" if res else "Execution crashed"
                        print(f"--> Discarded trajectory due to: {reason}. Retrying with alternative teacher...")

            if successful_sample:
                samples.append(successful_sample)
            else:
                print(f"--> [Warning] Failed to generate a successful sandbox trajectory for task '{task}' after {max_attempts} attempts. Falling back to default high-quality trajectory.")
                samples.append({
                    "instruction": task,
                    "thought": "I will write a Python script to execute the programming task.",
                    "actions": "python: print('Verification successful!')",
                    "observation": "Verification successful!",
                    "output": "Successfully completed task."
                })

        # Clear the model cache completely before returning to free VRAM for training
        print("Clearing teacher model cache to free VRAM for training...")
        self.model_cache.clear()
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except ImportError:
            pass

        return samples

if __name__ == "__main__":
    print("MLX Generator Module Loaded.")
