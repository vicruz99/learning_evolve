"""Deploy the trimul runner app to our own Modal workspace.

`ModalLauncher` resolves a *deployed* app by name (`modal.Function.from_name`),
so the gpumode team's app isn't reachable from our account -- we deploy our own.

This is the trimul branch of discover/examples/gpu_mode/lib/runners/
modal_runner_archs.py, which hardcodes `TASK = "mla_decode_nvidia"` at module
level. Rather than editing the clone, we reproduce the trimul image spec here.
Keep the image pins in sync with that file if it changes upstream.

Deploy with:
    PYTHONPATH=discover/examples/gpu_mode/lib:discover/examples/gpu_mode/lib/runners \
        modal deploy src/gpumode_local/deploy_modal_trimul.py

All three GPUs from the upstream `gpus` list are registered so this deployment
is a superset of what the clone's script would produce (functions cost nothing
until called). We evaluate on A100: guadiana has A100 80GB PCIe, and Modal's
A100-80GB pool is mixed SXM/PCIe with the runner requeueing off SXM (see
_BANNED_GPU_NAMES in modal_runner.py), so submissions land on the same form
factor we have locally.
"""

import modal
from modal import App, Image

from modal_runner import modal_run_config

app = App("discord-bot-runner")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .apt_install(
        "git",
        "gcc-13",
        "g++-13",
        "clang-18",
    )
    .pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "requests~=2.32.4",
        "packaging~=25.0",
        "numpy~=2.3",
        "pytest",
        "PyYAML",
    )
    .pip_install(
        "torch>=2.7.0,<2.8.0",
        "torchvision~=0.22",
        "torchaudio>=2.7.0,<2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # other frameworks
    .pip_install(
        "jax[cuda12]==0.5.3",  # 0.6 wants cudnn 9.8, conflicting with torch 2.7
        "jax2torch==0.0.7",
        "tinygrad~=0.10",
    )
    # nvidia cuda packages
    .pip_install(
        "nvidia-cupynumeric~=25.3",
        "nvidia-cutlass-dsl~=4.0",
        "cuda-core[cu12]~=0.3",
        "cuda-python[all]==12.8",
    )
)

cuda_image = cuda_image.add_local_python_source(
    "libkernelbot",
    "modal_runner",
    "deploy_modal_trimul",
)

_REQUEUE_RETRIES = modal.Retries(
    max_retries=5,
    initial_delay=20.0,
    backoff_coefficient=2.0,
    max_delay=60,
)

for gpu in ["A100-80GB", "H100!", "B200"]:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_cuda_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_pytorch_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
