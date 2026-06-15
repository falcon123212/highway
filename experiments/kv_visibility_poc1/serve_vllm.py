import os
import subprocess
import time
import urllib.request
import json
import argparse
import sys
from pathlib import Path

def check_server_ready(port: int) -> bool:
    try:
        url = f"http://localhost:{port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if "data" in data and len(data["data"]) > 0:
                    return True
    except Exception:
        pass
    return False

def start_server(model: str, port: int, max_model_len: int, gpu_util: float, dtype: str) -> subprocess.Popen:
    project_root = Path(__file__).resolve().parents[2]
    script_win = project_root / "experiments" / "kv_visibility_poc1" / "start_vllm.sh"
    script_wsl = _windows_path_to_wsl(script_win)

    # Ensure the script is executable in WSL
    subprocess.run(["wsl", "chmod", "+x", script_wsl], stdin=subprocess.DEVNULL)

    print(f"Launching vLLM server via {script_wsl}")
    log_file = open(script_win.with_name("vllm_server.log"), "w")

    wsl_cmd = [
        "wsl", "bash", script_wsl,
        model, str(port), str(max_model_len), f"{gpu_util:.2f}", dtype
    ]
    process = subprocess.Popen(wsl_cmd, stdout=log_file, stderr=log_file, stdin=subprocess.DEVNULL)
    return process


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if drive:
        parts = resolved.parts[1:]
        return f"/mnt/{drive}/" + "/".join(parts)
    return str(resolved).replace("\\", "/")

def main():
    parser = argparse.ArgumentParser(description="Serve vLLM inside WSL2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=55000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    
    # Check if a server is already running
    if check_server_ready(args.port):
        print(f"vLLM server is already running and ready on port {args.port}.")
        sys.exit(0)
        
    process = start_server(
        args.model, args.port, args.max_model_len, 
        args.gpu_memory_utilization, args.dtype
    )
    
    print("Waiting for vLLM server to start (polling http://localhost:8000/v1/models)...")
    start_time = time.time()
    ready = False
    
    while time.time() - start_time < args.timeout_seconds:
        if check_server_ready(args.port):
            ready = True
            break
        # Check if the process crashed
        poll = process.poll()
        if poll is not None:
            print(f"Error: vLLM server process terminated with exit code {poll}.")
            # Print last 20 lines of log file
            if os.path.exists("experiments/kv_visibility_poc1/vllm_server.log"):
                with open("experiments/kv_visibility_poc1/vllm_server.log", "r") as f:
                    lines = f.readlines()
                    print("Last 20 lines of vllm_server.log:")
                    for line in lines[-20:]:
                        print(line, end="")
            sys.exit(1)
        time.sleep(2)
        
    if ready:
        print(f"\n==========================================")
        print(f"vLLM server is ready on port {args.port}!")
        print(f"Process PID (Windows side wrapper): {process.pid}")
        print(f"Logs are writing to: experiments/kv_visibility_poc1/vllm_server.log")
        print(f"==========================================")
    else:
        print("Timeout reached. vLLM server is not responding.")
        # Kill process
        process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()


