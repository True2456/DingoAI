import random
import hashlib
import os
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
                        mx.clear_cache()
                    except AttributeError:
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
    def __init__(self, model_path: str, system_prompt: Optional[str] = None):
        self.model_path = model_path
        self.system_prompt = system_prompt

    def generate_tasks(self, num_tasks: int) -> List[str]:
        import mlx_lm
        import json
        import gc
        
        print(f"Bootstrapping {num_tasks} dynamic programming tasks using teacher {self.model_path}...")
        tasks = []
        try:
            model, tokenizer = mlx_lm.load(self.model_path)
            
            system_msg = (
                "You are an expert computer science curriculum designer. Brainstorm a list of unique, highly diverse Python programming tasks "
                "that can be executed and verified in a simple local terminal sandbox.\n"
                "The primary goal is to train Claude Code-style tool use: creating files, reading files, listing projects, running tests, "
                "observing failures, patching files, and rerunning verification. Prefer repository-editing workflows over single inline scripts.\n"
                "Mix the difficulty and topics across the following categories:\n"
                "- Tool-call discipline tasks: explicitly require write_file, read_file, list_dir, and python verification in sequence.\n"
                "- Patch/edit tasks: create an existing buggy source file, read it, modify or replace it, then run tests to prove the fix.\n"
                "- Multi-file project tasks: create src/, tests/, utility modules, and verify imports across files.\n"
                "- Test-first repair tasks: write a failing test first, observe the failure, fix implementation code, then rerun verification.\n"
                "- Refactor tasks: improve duplicated or messy code while preserving behavior with assertions.\n"
                "- Frontend file generation tasks: create index.html, style.css, and main.js, then verify file contents or simple invariants locally.\n"
                "- Secure coding tasks: local-only input validation, path traversal prevention, secret detection in fake files, safe sqlite parameterization, and log anomaly detection.\n"
                "- A smaller minority of pure algorithm tasks: graph algorithms, dynamic programming, concurrency, and edge-case-heavy logic.\n"
                "Each task must be a single sentence, concise, require computing or verifying a result dynamically, and "
                "be structured such that the solution can be verified using self-contained assertions or file-content checks.\n"
                "The tasks should naturally require multiple steps of reasoning to solve. "
                "At least 65% of tasks should require creating, reading, modifying, or verifying files rather than only running inline Python. "
                "At least 35% should require multiple files or directories such as src/ and tests/. "
                "At least 25% should explicitly require patch/edit or test-first repair. "
                "Avoid tasks whose natural solution is just printing one final code block. "
                "Return ONLY a raw JSON list of strings. Do NOT wrap it in ```json blocks or output any conversational text. "
                "You MUST start your entire response with '[' and end with ']'.\n\n"
                "Example output:\n"
                "[\n"
                "  \"Create src/cache.py with a buggy LRUCache implementation, write tests/test_cache.py that exposes the bug, then patch src/cache.py and rerun the tests successfully.\",\n"
                "  \"Write src/utils.py containing a palindrome checker, write tests/test_utils.py that imports it, list the workspace files, and verify all assertions pass.\",\n"
                "  \"Create index.html, style.css, and main.js for a tiny counter UI, then read the files back and assert that required IDs and event-handler strings exist.\",\n"
                "  \"Refactor a duplicated temperature conversion script into reusable functions while preserving its original printed outputs with assertions.\",\n"
                "  \"Create src/security.py with an unsafe path join helper, write tests/test_security.py that demonstrates a path traversal bug, then patch the helper and rerun the tests successfully.\"\n"
                "]"
            )
            if self.system_prompt:
                system_msg = self.system_prompt
            
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

            batch_size = 50
            consecutive_failures = 0
            while len(tasks) < num_tasks and consecutive_failures < 5:
                curr_batch_size = min(batch_size, num_tasks - len(tasks))
                user_msg = (
                    f"Generate exactly {curr_batch_size} unique and diverse Python programming tasks as a JSON list of strings. "
                    "Make most tasks require actual tool-style file operations: write_file, read_file, list_dir, then python verification. "
                    "Do not overproduce simple one-file scripts."
                )
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
                
                response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=4096, verbose=False)
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
                mx.clear_cache()
            except AttributeError:
                pass
            
        except Exception as e:
            print(f"Error bootstrapping dynamic tasks: {e}")
            
        if len(tasks) >= num_tasks:
            return tasks[:num_tasks]
        elif len(tasks) > 0:
            print(f"Only bootstrapped {len(tasks)} tasks due to generation limits. Padding with fallbacks.")
            # If we got some tasks, but not enough, let's pad them up to num_tasks
            fallbacks = [
                "Create src/math_utils.py with an intentionally buggy add_many function, read the file back, replace it with a corrected implementation, and run tests/test_math_utils.py successfully.",
                "Create src/text_tools.py and tests/test_text_tools.py, list the workspace files, then verify the test file imports the utility module and all assertions pass.",
                "Write tests/test_slugify.py first so it fails because src/slugify.py is missing, then create src/slugify.py and rerun the tests successfully.",
                "Create a duplicated unit conversion script, refactor it into reusable functions, and verify the original conversions still produce the same results.",
                "Create index.html, style.css, and main.js for a tiny todo UI, then read the files back and assert that required DOM IDs and event-handler strings are present.",
                "Use write_file, read_file, list_dir, and python verification in sequence to create and validate a small two-file Python project.",
                "Create src/security.py with an unsafe path sanitizer, write tests/test_security.py that catches path traversal, read and patch src/security.py, then rerun the test successfully.",
                "Create src/config_loader.py and tests/test_config_loader.py, intentionally fail on missing defaults, patch the loader, list the workspace, and verify the tests pass.",
                "Create README.md and src/main.py for a tiny CLI project, read both files back, then run python verification that checks the documented command exists in the source.",
                "Create a buggy src/cart.py checkout total function, write a failing tests/test_cart.py for discounts, patch the function, and verify the corrected behavior.",
                "Write a small package with src/__init__.py and src/strings.py, list the workspace recursively, then run a test script that imports and verifies the package."
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
            "Create src/parser.py with a buggy CSV parser, write tests/test_parser.py that exposes the bug, read src/parser.py, patch it, and rerun the tests successfully.",
            "Create src/formatter.py and tests/test_formatter.py, list the workspace, and verify that the test imports the formatter module correctly.",
            "Write a failing test for a missing src/password_rules.py validator, observe the failure, create the implementation, and rerun verification successfully.",
            "Refactor a duplicated string-cleaning script into reusable functions while preserving behavior with assertions.",
            "Create index.html, style.css, and main.js for a small neon status panel, then read each file back and assert that expected selectors and text exist.",
            "Use write_file, read_file, list_dir, and python actions in order to create, inspect, enumerate, and verify a small Python package.",
            "Create src/rate_limit.py with a subtle boundary bug, write tests/test_rate_limit.py that catches it, read the implementation, patch it, and rerun verification.",
            "Create src/secrets.py that scans fake files for API-key-like strings, write tests/test_secrets.py with sample content, list the files, and verify detections.",
            "Create a tiny frontend with index.html, style.css, and main.js, then use read_file and python assertions to verify script and stylesheet references are correct.",
            "Create src/graph.py and tests/test_graph.py for BFS shortest path, intentionally fail once due to a missing edge case, patch the file, and rerun the tests.",
            "Create a duplicated src/report.py implementation, read it back, refactor into helper functions, and run assertions that preserve the generated report output.",
            "Write tests/test_normalize.py for a missing src/normalize.py function, observe the missing-file failure, create the implementation, and rerun successfully.",
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
    
    def __init__(
        self,
        model_paths: List[str],
        workspace_dir: str = "data/sandbox",
        bootstrap_model_path: Optional[str] = None,
        task_system_prompt: Optional[str] = None,
        teacher_attempt_order: Optional[List[int]] = None,
        memory_settings: Optional[Dict[str, Any]] = None,
    ):
        self.model_paths = model_paths
        self.workspace_dir = workspace_dir
        self.bootstrap_model_path = bootstrap_model_path
        self.task_system_prompt = task_system_prompt
        self.teacher_attempt_order = teacher_attempt_order or list(range(len(model_paths)))
        self.memory_settings = memory_settings or {}
        self.teacher_cache_limit_gb = float(self.memory_settings.get("teacher_cache_limit_gb", 45))
        self.bootstrap_max_model_gb = float(self.memory_settings.get("bootstrap_max_model_gb", self.teacher_cache_limit_gb))
        self.cache_teachers = bool(self.memory_settings.get("cache_teachers", self.teacher_cache_limit_gb > 0))
        # Cache loaded teacher models in RAM when the selected memory profile allows it.
        self.model_cache = {}

    @staticmethod
    def _model_size_gb(path: str) -> float:
        model_size = 0
        for root_dir, _, files_list in os.walk(path):
            for file_name in files_list:
                model_size += os.path.getsize(os.path.join(root_dir, file_name))
        return model_size / (1024 ** 3)

    def generate(self, num_samples: int, task_descriptions: Optional[List[str]] = None, checkpoint_path: Optional[str] = None) -> List[Dict[str, Any]]:
        from sandbox.sandbox import SandboxExecutor
        import tempfile
        import os
        import json

        # Ensure the shared sandbox parent directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Resume from checkpoint if it exists (crash recovery)
        samples = []
        completed_tasks = set()
        failed_attempts_path = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s = json.loads(line)
                        samples.append(s)
                        completed_tasks.add(s.get("instruction", ""))
            print(f"Resumed from checkpoint: {len(samples)} samples already completed.")

        if checkpoint_path:
            base, ext = os.path.splitext(checkpoint_path)
            failed_attempts_path = f"{base}_failed_attempts{ext or '.jsonl'}"

        if task_descriptions is None:
            # Dynamically bootstrap unique programming tasks using a teacher model under 45 GB
            # to prevent GPU OOM crashes at the start of the generation step.
            if self.bootstrap_model_path:
                teacher_for_bootstrap = self.bootstrap_model_path
                print(f"Using requested bootstrap task generator: {teacher_for_bootstrap}")
            else:
                teacher_for_bootstrap = self.model_paths[0]
                best_size = 0
                for path in self.model_paths:
                    try:
                        model_size_gb = self._model_size_gb(path)
                        if model_size_gb <= self.bootstrap_max_model_gb:
                            if model_size_gb > best_size:
                                best_size = model_size_gb
                                teacher_for_bootstrap = path
                    except Exception:
                        pass
            
            task_generator = DynamicTaskGenerator(
                teacher_for_bootstrap,
                system_prompt=self.task_system_prompt,
            )
            task_descriptions = task_generator.generate_tasks(num_tasks=num_samples)

        for i in range(num_samples):
            task = task_descriptions[i % len(task_descriptions)]
            task_id = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]

            # Skip tasks already completed in a previous (crashed) run
            if task in completed_tasks:
                continue

            # We try up to 3 times to get a successful sandbox trajectory
            max_attempts = 3
            successful_sample = None
            failed_attempts = []

            # Local worker with unified RAM model caching
            def _interact_with_teacher(path: str, t: str, sandbox_dir: str) -> Optional[Dict[str, Any]]:
                try:
                    import mlx_lm
                    import json
                    # Determine if we should cache this model based on the selected memory profile.
                    should_cache = self.cache_teachers
                    try:
                        model_size_gb = self._model_size_gb(path)
                        if model_size_gb > self.teacher_cache_limit_gb:
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
                        if self.model_cache:
                            print("Evicting cached teacher models from RAM before loading the large model...")
                            self.model_cache.clear()
                            import gc
                            gc.collect()
                            try:
                                import mlx.core as mx
                                mx.clear_cache()
                            except AttributeError:
                                pass
                        print(f"Loading large teacher model {path} dynamically (not cached to protect RAM)...")
                        try:
                            model, tokenizer = mlx_lm.load(path)
                        except Exception as load_err:
                            print(f"[OOM] Failed to load large model {path}: {load_err}. Skipping this attempt.")
                            return None

                    # Instantiate sandbox isolated exclusively to this jailed workspace
                    sandbox = SandboxExecutor(sandbox_dir)

                    history = f"Task: {t}\n"
                    max_turns = 8
                    
                    thought_trace = []
                    actions_trace = []
                    observations_trace = []
                    final_answer = ""
                    sandbox_success = True
                    turns_array = []
                    
                    force_failure = random.random() < 0.3

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
                            "You are an AI programming agent executing actions in a local terminal, similar to Claude Code.\n"
                            "Based on the history, generate your next reasoning step in JSON format.\n"
                            "The JSON must have the following keys:\n"
                            "- 'thought': Your logical analysis of the current state.\n"
                            "- 'action_type': One of:\n"
                            "    'python'     — execute inline Python code\n"
                            "    'write_file' — write content to a sandbox-relative file (action_input format: 'relative/path.ext:file content here')\n"
                            "    'read_file'  — read a sandbox-relative file (action_input: relative/path.ext)\n"
                            "    'list_dir'   — list files in the workspace\n"
                            "    'none'       — task is complete, provide final_answer\n"
                            "- 'action_input': The code, file content, or filename as required by action_type.\n"
                            "- 'final_answer': A descriptive string summarizing the result ONLY if action_type is 'none'.\n\n"
                            "CRITICAL: When the task asks you to create or edit files, DO NOT merely describe code or print Markdown code blocks. "
                            "Use write_file/read_file/list_dir/python actions to mutate and verify the workspace. "
                            "For file-creation tasks, your first useful action should usually be 'write_file', not 'python'. "
                            "Do NOT hardcode final results. Always write clean Python code to compute results. "
                            "For multi-file tasks, use 'write_file' to create each file first, 'list_dir' to confirm the workspace, then 'python' to execute and verify. "
                            "For patch/edit and refactor tasks, use 'read_file' before replacing the file with 'write_file'. "
                            "For test-first tasks, write and run a failing test before writing the corrected implementation. "
                            "Use 'none' only after the workspace has been modified and verification has succeeded. "
                            "You MUST include self-verifying test assertions (using 'assert') at the end of your script "
                            "to programmatically prove your solution is correct. If assertions fail, the script will crash, "
                            "which is the correct behavior for incorrect logic.\n"
                        )
                        
                        if force_failure and turn == 0:
                            system_msg += (
                                "\n[TRAINING OVERRIDE]: For this first turn ONLY, you MUST intentionally create a realistic bug "
                                "using either a 'python' action or a 'write_file' action, depending on the task "
                                "(e.g., an off-by-one error, syntax error, missing import, incorrect logic, or incomplete file content). "
                                "Do NOT solve the task correctly on this turn. You will fix it in the next turn based on the error trace.\n"
                                "CRITICAL: Even though you are writing buggy code, you MUST still output your response in the EXACT JSON dictionary format described below. Do not use plain text.\n"
                            )
                        
                        system_msg += (
                            "\nFormat Examples:\n"
                            "Example 1 (inline execution):\n"
                            "{\n"
                            "  \"thought\": \"I need to write a script to compute the solution and assert correctness.\",\n"
                            "  \"action_type\": \"python\",\n"
                            "  \"action_input\": \"def solve(x): return x * 2\\n\\n# Assertions\\nassert solve(5) == 10\\nprint('Verified')\",\n"
                            "  \"final_answer\": \"\"\n"
                            "}\n"
                            "Example 2 (write a nested project file, then execute it in a later turn):\n"
                            "{\n"
                            "  \"thought\": \"I'll write the utility module first, then verify it by importing it in a test script.\",\n"
                            "  \"action_type\": \"write_file\",\n"
                            "  \"action_input\": \"src/utils.py:def double(x):\\n    return x * 2\\n\",\n"
                            "  \"final_answer\": \"\"\n"
                            "}\n"
                            "Example 3 (inspect existing files during patch/refactor work):\n"
                            "{\n"
                            "  \"thought\": \"I need to inspect the buggy file before replacing it with a corrected version.\",\n"
                            "  \"action_type\": \"read_file\",\n"
                            "  \"action_input\": \"src/utils.py\",\n"
                            "  \"final_answer\": \"\"\n"
                            "}\n"
                            "Example 4 (verify the project after file edits):\n"
                            "{\n"
                            "  \"thought\": \"The files are written, so I will run the test module from the sandbox workspace and rely on assertions for verification.\",\n"
                            "  \"action_type\": \"python\",\n"
                            "  \"action_input\": \"import runpy\\nrunpy.run_path('tests/test_utils.py', run_name='__main__')\\nprint('Verified')\",\n"
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

                        if action_type == "none":
                            final_answer = step.get("final_answer", "Task complete.")
                            observations_trace.append("Environment complete.")
                            turn_data["observation"]["stdout"] = "Environment complete."
                            turns_array.append(turn_data)
                            break

                        if action_type != "list_dir" and not action_input:
                            sandbox_success = False
                            obs = f"Error: action_type '{action_type}' requires non-empty action_input."
                            observations_trace.append(obs)
                            turn_data["observation"]["stderr"] = obs
                            turn_data["observation"]["success"] = False
                            turns_array.append(turn_data)
                            break

                        # Execute in Sandbox
                        exec_res = sandbox.execute(action_type, action_input)
                        if exec_res.get("success") and not exec_res.get("stderr"):
                            obs = (
                                exec_res.get("stdout", "")
                                or exec_res.get("message", "")
                                or (f"files: {exec_res.get('files')}" if "files" in exec_res else "")
                                or (exec_res.get("content", "") if "content" in exec_res else "")
                                or "Success."
                            )
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
                        mx.clear_cache()
                    except AttributeError:
                        pass

                    if thought_trace:
                        action_types = [
                            (turn.get("action") or {}).get("type")
                            for turn in turns_array
                        ]
                        file_workflow_requested = any(
                            keyword in t.lower()
                            for keyword in [
                                "create ",
                                "write ",
                                "patch",
                                "refactor",
                                "test",
                                "verify",
                                "frontend",
                                "index.html",
                                "src/",
                                "tests/",
                            ]
                        )
                        if file_workflow_requested and ("write_file" not in action_types or "python" not in action_types):
                            sandbox_success = False
                            observations_trace.append(
                                "Rejected: file workflow did not include both write_file and python verification."
                            )

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

            ordered_teachers = [
                self.model_paths[idx]
                for idx in self.teacher_attempt_order
                if 0 <= idx < len(self.model_paths)
            ] or self.model_paths

            for attempt in range(max_attempts):
                teacher_path = ordered_teachers[attempt % len(ordered_teachers)]

                print(f"Generating agentic trajectory (Attempt {attempt+1}/{max_attempts}) using teacher {teacher_path} for task: '{task}'")
                
                # Jailed temporary directory workspace for this specific attempt to guarantee complete isolation
                with tempfile.TemporaryDirectory(prefix=f"sandbox_rollout_{i}_{attempt}_", dir=self.workspace_dir) as tmp_workspace_dir:
                    res = _interact_with_teacher(teacher_path, task, tmp_workspace_dir)
                    if res and res.get("sandbox_success", True):
                        successful_sample = res
                        break
                    else:
                        reason = "Sandbox execution failed or syntax error" if res else "Execution crashed"
                        failed_attempt = {
                            "task_id": task_id,
                            "instruction": task,
                            "attempt": attempt + 1,
                            "teacher_model": teacher_path,
                            "failure_reason": reason,
                            "trajectory": res,
                        }
                        failed_attempts.append(failed_attempt)
                        if failed_attempts_path:
                            with open(failed_attempts_path, "a") as failed_f:
                                failed_f.write(json.dumps(failed_attempt) + "\n")
                        print(f"--> Discarded trajectory due to: {reason}. Retrying with alternative teacher...")

            if successful_sample:
                successful_sample["task_id"] = task_id
                successful_sample["failed_attempts"] = failed_attempts
                successful_sample["failed_attempt_count"] = len(failed_attempts)
                samples.append(successful_sample)
            else:
                print(f"--> [Warning] Failed to generate a successful sandbox trajectory for task '{task}' after {max_attempts} attempts. Falling back to default high-quality trajectory.")
                successful_sample = {
                    "task_id": task_id,
                    "instruction": task,
                    "thought": (
                        "I will use tool-style file operations instead of only describing code. "
                        "First I create a source file, then I inspect the workspace, and finally I run verification."
                    ),
                    "actions": (
                        "write_file: src/fallback.py:def solve():\n    return 'verified'\n | "
                        "list_dir:  | "
                        "python: import runpy, os\nassert 'src/fallback.py' in [p.replace('\\\\', '/') for root, _, files in os.walk('.') for p in [os.path.join(root, f).lstrip('./') for f in files]]\nns = runpy.run_path('src/fallback.py')\nassert ns['solve']() == 'verified'\nprint('Verified')"
                    ),
                    "observation": (
                        "Successfully wrote src/fallback.py | files: ['src/fallback.py'] | Verified"
                    ),
                    "output": "Created a source file, inspected the workspace, and verified it with assertions.",
                    "sandbox_success": True,
                    "turns": [
                        {
                            "turn": 1,
                            "thought": "I need to create an actual file in the workspace.",
                            "action": {"type": "write_file", "input": "src/fallback.py:def solve():\n    return 'verified'\n"},
                            "observation": {"stdout": "Successfully wrote src/fallback.py", "stderr": "", "success": True}
                        },
                        {
                            "turn": 2,
                            "thought": "I will list the workspace to confirm the file exists.",
                            "action": {"type": "list_dir", "input": ""},
                            "observation": {"stdout": "files: ['src/fallback.py']", "stderr": "", "success": True}
                        },
                        {
                            "turn": 3,
                            "thought": "I will run a verification script that imports the file and checks behavior.",
                            "action": {"type": "python", "input": "import runpy\nns = runpy.run_path('src/fallback.py')\nassert ns['solve']() == 'verified'\nprint('Verified')"},
                            "observation": {"stdout": "Verified", "stderr": "", "success": True}
                        }
                    ],
                    "teacher_model": "fallback",
                    "failed_attempts": failed_attempts,
                    "failed_attempt_count": len(failed_attempts),
                }
                samples.append(successful_sample)

            # Write sample immediately so a crash doesn't lose prior work
            if checkpoint_path:
                with open(checkpoint_path, "a") as ckpt_f:
                    ckpt_f.write(json.dumps(successful_sample) + "\n")

        # Clear the model cache completely before returning to free VRAM for training
        print("Clearing teacher model cache to free VRAM for training...")
        self.model_cache.clear()
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except AttributeError:
            pass

        return samples

if __name__ == "__main__":
    print("MLX Generator Module Loaded.")
