import os
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

    def execute(self, action_type: str, command: str) -> Dict[str, Any]:
        """
        Executes a sandboxed action.
        
        Supported actions:
        - 'python': Runs inline Python code.
        - 'write_file': Writes content to a file (command is formatted as file_path:content).
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
            return {
                "success": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.returncode
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

    def _write_file(self, content_str: str) -> Dict[str, Any]:
        try:
            # Format: 'filename:content'
            if ":" not in content_str:
                return {"success": False, "error": "Invalid write format. Use 'relative/path:content'"}
            
            parts = content_str.split(":", 1)
            filename = parts[0].strip()
            content = parts[1]

            file_path = self._safe_path(filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            return {"success": True, "message": f"Successfully wrote {len(content)} chars to {filename}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_file(self, filename: str) -> Dict[str, Any]:
        try:
            filename = filename.strip()
            file_path = self._safe_path(filename)
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File '{filename}' not found."}
            
            with open(file_path, "r") as f:
                content = f.read()
            return {"success": True, "content": content}
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
