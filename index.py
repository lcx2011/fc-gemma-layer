import os
import json
import signal
import time
import subprocess
import sys
from threading import Thread

model = None
MODEL_PATH = "/tmp/gemma-4-E2B-it-Q4_K_M.gguf"

def install_deps():
    """全部从阿里云镜像安装"""
    print("Installing llama-cpp-python...", flush=True)
    subprocess.run([
        "pip", "install",
        "-i", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com",
        "llama-cpp-python"
    ], check=True)
    
    print("Installing modelscope...", flush=True)
    subprocess.run([
        "pip", "install",
        "-i", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com",
        "modelscope"
    ], check=True)
    
    print("Dependencies installed!", flush=True)

def download_model():
    """从 ModelScope 下载模型，实时打印进度"""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already cached: {size_mb:.1f} MB", flush=True)
        return
    
    print("Downloading model from ModelScope (1.8 GB)...", flush=True)
    print("Progress will update every 10 seconds...", flush=True)
    
    from modelscope.hub.file_download import model_file_download
    
    # 用 wget 替代，可以实时看进度
    model_url = "https://hf-mirror.com/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
    
    print(f"Downloading from: hf-mirror.com", flush=True)
    print("=" * 50, flush=True)
    
    # 使用 wget 并实时输出进度
    result = subprocess.run([
        "wget",
        "--progress=bar:force",   # 强制显示进度条
        "--show-progress",
        "-O", MODEL_PATH,
        model_url
    ], stderr=subprocess.STDOUT, stdout=sys.stderr)
    
    if result.returncode != 0:
        raise Exception("Download failed")
    
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print("=" * 50, flush=True)
    print(f"Download complete! Size: {size_mb:.1f} MB", flush=True)

def load_model():
    global model
    from llama_cpp import Llama
    
    print("Loading model into memory...", flush=True)
    start = time.time()
    
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=1,
        n_batch=512,
        verbose=False
    )
    
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s!", flush=True)

def shutdown():
    time.sleep(0.5)
    print("Shutting down to stop billing...", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

# ========== 冷启动入口 ==========
print("=" * 50, flush=True)
print("Cold start begin...", flush=True)
t0 = time.time()

install_deps()
download_model()
load_model()

print(f"Total cold start time: {time.time() - t0:.1f}s", flush=True)
print("=" * 50, flush=True)

def handler(event, context):
    try:
        if isinstance(event, bytes):
            body = json.loads(event.decode("utf-8"))
        elif isinstance(event, str):
            body = json.loads(event)
        else:
            body = event
    except:
        body = {"prompt": str(event)}
    
    prompt = body.get("prompt", body.get("text", ""))
    max_tokens = body.get("max_tokens", 128)
    
    if not prompt:
        return {"statusCode": 400, "body": json.dumps({"error": "prompt required"})}
    
    print(f"Generating: {prompt[:50]}...", flush=True)
    t1 = time.time()
    
    messages = [{"role": "user", "content": prompt}]
    
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    
    response_text = output["choices"][0]["message"]["content"]
    print(f"Generated {len(response_text)} chars in {time.time() - t1:.1f}s", flush=True)
    
    Thread(target=shutdown, daemon=True).start()
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_text}, ensure_ascii=False)
    }