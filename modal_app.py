#!/usr/bin/env python3
"""Modal deployment of the NEA1 inference service (T4 GPU).

Runs the SAME server.py as the Render deployment — same endpoints,
protocol, and auth — on a T4 GPU with the serving checkpoint baked into
the image. Scale-to-zero: a container starts on request, stays warm for
the scaledown window, then shuts off, so the Starter plan's monthly
compute credits only burn while the service is actually up.

Deliberately NOT kept warm by the keepalive worker (that worker still
pings Render, the fallback): a 10-minute ping cadence would hold a GPU
container open around the clock and drain the credits.

Deploy (from this directory, with MODAL_TOKEN_ID/MODAL_TOKEN_SECRET set):
    modal deploy modal_app.py
Serving URL:
    https://<workspace>--neuralai-nae1-serve.modal.run
"""
import modal

app = modal.App("neuralai-nae1")

HF_REPO = "Subject-Emu-5259/NeuralAI-Nae1"
HF_SUBDIR = "sft_v9_lora_step12000_bf16"
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json",
               "tokenizer_config.json")


def _bake_model():
    """Image build step: fetch the serving checkpoint into /model."""
    import pathlib
    import shutil

    from huggingface_hub import hf_hub_download

    for name in MODEL_FILES:
        hf_hub_download(repo_id=HF_REPO, filename=f"{HF_SUBDIR}/{name}",
                        local_dir="/model-root")
    src = pathlib.Path("/model-root") / HF_SUBDIR
    dst = pathlib.Path("/model")
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy(src / name, dst / name)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu121 "
        "\"torch==2.5.1\"",
        "pip install \"transformers==4.48.3\" \"tokenizers==0.21.0\" "
        "\"huggingface_hub==0.26.0\" \"jinja2==3.1.4\" \"numpy\" "
        "\"safetensors\"",
    )
    .run_function(_bake_model)
    .add_local_file("server.py", "/app/server.py")
    .add_local_file("nae1_model.py", "/app/nae1_model.py")
    .add_local_file("nae1_config.py", "/app/nae1_config.py")
)


@app.function(
    image=image,
    # CPU serving: Modal gates GPU functions behind a payment method on
    # file; CPU containers need none and still outrun Render's free tier
    # by a wide margin. 4 cores + bf16 matches the production CPU recipe.
    cpu=4.0,
    memory=4096,
    secrets=[modal.Secret.from_name("nae1-inference")],
    env={
        "PORT": "8080",
        "DEVICE": "cpu",
        "DTYPE": "bfloat16",
        "BLOCKING_LOAD": "1",
        "MODEL_DIR": "/model",
        "DATA_DIR": "/app/data",
        "PYTHONUNBUFFERED": "1",
    },
    scaledown_window=300,
    max_containers=2,
    timeout=1200,
)
@modal.web_server(8080, startup_timeout=600)
def serve():
    import subprocess

    subprocess.Popen(["python", "/app/server.py"], cwd="/app")
