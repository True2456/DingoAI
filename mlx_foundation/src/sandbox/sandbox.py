import os
import json
import subprocess
import tempfile
from typing import Dict, Any

class SandboxExecutor:
    """
    A safe, localized execution environment to run Python scripts and commands
    and capture real environmental outputs for agentic training loops.
    """

    def __init__(self, workspace_dir: str = "data/sandbox"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def execute(self, action_type: str, command: Any) -> Dict[str, Any]:
        """
        Executes a sandboxed action.
        
        Supported actions:
        - 'python': Runs inline Python code.
        - 'write_file': Writes content to a file.
          Preferred command format: {"path": "relative/path.ext", "content": "..."}.
          Legacy command format: "relative/path.ext:content".
        - 'read_file': Reads content of a file (command is file_path).
        - 'list_dir': Lists files in the sandbox workspace.
        """
        if action_type == "python":
            return self._run_python(command)
        elif action_type == "write_file":
            return self._write_file(command)
        elif action_type == "read_file":
            return self._read_file(command)
        elif action_type == "list_dir":
            return self._list_dir()
        elif action_type == "edit":
            return self._edit_file(command)
        elif action_type == "bash":
            return self._run_bash(command)
        else:
            return {
                "success": False,
                "error": f"Unsupported sandboxed action type: {action_type}"
            }

    def _safe_path(self, relative_path: str) -> str:
        """Resolve a sandbox-relative path without allowing escapes."""
        cleaned = relative_path.strip().lstrip("/\\")
        full_path = os.path.abspath(os.path.join(self.workspace_dir, cleaned))
        if not full_path.startswith(self.workspace_dir + os.sep) and full_path != self.workspace_dir:
            raise ValueError(f"Path escapes sandbox: {relative_path}")
        return full_path

    def _looks_like_failed_test_output(self, stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}"
        failure_markers = [
            "FAILED [",
            " failed, ",
            " failures, ",
            " errors, ",
            "AssertionError",
            "Traceback (most recent call last):",
        ]
        return any(marker in combined for marker in failure_markers)

    def _looks_like_passing_test_output(self, stdout: str, stderr: str) -> bool:
        """unittest often prints progress to stderr even when all tests pass."""
        combined = f"{stdout}\n{stderr}"
        if self._looks_like_failed_test_output(stdout, stderr):
            return False
        return (
            "OK" in combined
            or "All tests passed" in combined
            or "Verification successful" in combined
            or "Verified" in combined
        )

    def _run_python(self, code: str) -> Dict[str, Any]:
        # Write inline code to a temporary file in the sandbox workspace
        with tempfile.NamedTemporaryFile(suffix=".py", dir=self.workspace_dir, delete=False, mode="w") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Run using the python interpreter from the current venv if possible
            python_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../venv/bin/python"))
            if not os.path.exists(python_bin):
                python_bin = "python3"

            res = subprocess.run(
                [python_bin, tmp_path],
                capture_output=True,
                text=True,
                cwd=self.workspace_dir,
                timeout=5 # Safe timeout limit
            )
            failed_test = self._looks_like_failed_test_output(res.stdout, res.stderr)
            passed_test = self._looks_like_passing_test_output(res.stdout, res.stderr)
            verification_passed = res.returncode == 0 and not failed_test
            expected_test_failure = failed_test and not verification_passed
            return {
                "success": verification_passed,
                "verification_passed": verification_passed,
                "expected_test_failure": expected_test_failure,
                "tests_passed_signal": passed_test,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out (limit: 5 seconds)."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _parse_write_file_input(self, command: Any) -> tuple[str, str] | tuple[None, None]:
        if isinstance(command, dict):
            filename = str(
                command.get("path") or command.get("file_path") or ""
            ).strip()
            content = command.get("content", "")
            return filename, str(content)

        if not isinstance(command, str):
            return None, None

        stripped = command.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    filename = str(parsed.get("path") or parsed.get("file_path") or "").strip()
                    content = parsed.get("content", "")
                    return filename, str(content)
            except json.JSONDecodeError:
                pass

        # Legacy format: 'filename:content'
        if ":" not in command:
            return None, None
        filename, content = command.split(":", 1)
        return filename.strip(), content

    def _write_file(self, command: Any) -> Dict[str, Any]:
        try:
            filename, content = self._parse_write_file_input(command)
            if filename is None:
                return {
                    "success": False,
                    "error": "Invalid write format. Use {'path': 'relative/path.ext', 'content': '...'}."
                }
            if not filename or "\n" in filename or "\r" in filename:
                return {
                    "success": False,
                    "error": "Invalid write path. Use action_input.path with a sandbox-relative filename."
                }
            if filename.endswith((".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", ".csv", ".xml")) is False and "/" not in filename:
                return {
                    "success": False,
                    "error": "Invalid write path. Include a filename with an extension or a sandbox-relative directory path."
                }

            file_path = self._safe_path(filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            return {"success": True, "message": f"Successfully wrote {len(content)} chars to {filename}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_file(self, filename: Any) -> Dict[str, Any]:
        try:
            if isinstance(filename, dict):
                filename = str(
                    filename.get("file_path") or filename.get("path") or ""
                )
            filename = str(filename).strip()
            file_path = self._safe_path(filename)
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File '{filename}' not found."}
            
            with open(file_path, "r") as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _edit_file(self, command: Any) -> Dict[str, Any]:
        """Search/replace patch (Claude Edit tool shape)."""
        try:
            if isinstance(command, dict):
                path = str(command.get("file_path") or command.get("path") or "").strip()
                old_string = command.get("old_string", "")
                new_string = command.get("new_string", "")
            else:
                return {"success": False, "error": "Edit requires a dict with file_path, old_string, new_string."}
            if not path or old_string is None or new_string is None:
                return {"success": False, "error": "Edit missing file_path, old_string, or new_string."}
            file_path = self._safe_path(path)
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File '{path}' not found."}
            with open(file_path, "r") as f:
                content = f.read()
            if old_string not in content:
                return {
                    "success": False,
                    "error": f"old_string not found in {path}.",
                }
            updated = content.replace(old_string, new_string, 1)
            with open(file_path, "w") as f:
                f.write(updated)
            return {
                "success": True,
                "message": f"Patched {path} ({len(old_string)} -> {len(new_string)} chars).",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_bash(self, command: Any) -> Dict[str, Any]:
        """Run a shell command in the sandbox workspace (Claude Bash)."""
        cmd = command
        if isinstance(command, dict):
            cmd = command.get("command") or command.get("cmd") or ""
        if not isinstance(cmd, str) or not cmd.strip():
            return {"success": False, "error": "Bash requires non-empty command string."}
        python_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../venv/bin/python"))
        env = os.environ.copy()
        env["PYTHONPATH"] = self.workspace_dir + (os.pathsep + env.get("PYTHONPATH", ""))
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.workspace_dir,
                timeout=30,
                env=env,
            )
            failed_test = self._looks_like_failed_test_output(res.stdout, res.stderr)
            passed_test = self._looks_like_passing_test_output(res.stdout, res.stderr)
            verification_passed = res.returncode == 0 and not failed_test
            expected_test_failure = failed_test and not verification_passed
            return {
                "success": verification_passed,
                "verification_passed": verification_passed,
                "expected_test_failure": expected_test_failure,
                "tests_passed_signal": passed_test,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Bash command timed out (30s)."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_dir(self) -> Dict[str, Any]:
        try:
            files = []
            for root, _, filenames in os.walk(self.workspace_dir):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    files.append(os.path.relpath(full_path, self.workspace_dir))
            return {"success": True, "files": files}
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    sandbox = SandboxExecutor()
    res = sandbox.execute("python", "print('Hello from sandboxed interpreter!')")
    print(res)
