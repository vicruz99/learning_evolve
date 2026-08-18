#!/usr/bin/env bash
# Does the grading interpreter actually have kernels for the card this job was given?
#
#   KPY=~/venvs/kernel-eval/bin/python ./check_gpu.sh
#
# Run this on the compute node BEFORE the first TriMul run. A cu126 build on a Blackwell
# card does not fail cleanly: sm_100 gencode comes only from CUDA 12.8, so the cu126 wheel's
# arch list stops at sm_90 and you get a wall of ptxas errors that read like bad candidates.
# The arch list alone is not proof either -- the last step compiles and runs a real Triton
# kernel, which is what the grader will do.
set -uo pipefail

KPY=${KPY:-$HOME/venvs/kernel-eval/bin/python}
[ -x "$KPY" ] || { echo "KPY=$KPY is not executable"; exit 1; }

echo "== the card the scheduler granted =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || true
echo
echo "== the grading stack: $KPY =="

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/stack.py" <<'STACK_EOF'
import sys
import torch, triton

if not torch.cuda.is_available():
    sys.exit("NO GPU  torch sees no CUDA device. The venv is not the problem -- you are on a\n"
             "        node without a card (a login node, or a job submitted without -gpu).\n"
             "        Get an allocation first: README step 0.")

props = torch.cuda.get_device_properties(0)
cc = torch.cuda.get_device_capability(0)
sm = f"sm_{cc[0]}{cc[1]}"
archs = torch.cuda.get_arch_list()

print(f"torch            {torch.__version__}")
print(f"triton           {triton.__version__}")
print(f"card             {props.name}")
print(f"capability       {sm}")
print(f"arch list        {archs}")
print(f"shared mem/block {props.shared_memory_per_block_optin}   <- must match the prompt's rules line")

ok = True
if triton.__version__ != "3.3.1":
    print(f"\nFAIL  triton is {triton.__version__}; the harness pins 3.3.1")
    ok = False
if not torch.__version__.startswith("2.7.1"):
    print(f"\nFAIL  torch is {torch.__version__}; the harness pins 2.7.1")
    ok = False
if sm not in archs:
    print(f"\nFAIL  this interpreter has no kernels for {sm}. On a B200 you need the cu128"
          f"\n      build -- sm_100 gencode only comes from CUDA 12.8.")
    ok = False
if sm == "sm_100" and "cu128" not in torch.__version__:
    print(f"\nFAIL  {sm} needs a cu128 wheel; this is {torch.__version__}")
    ok = False
sys.exit(0 if ok else 1)
STACK_EOF

if ! "$KPY" "$TMP/stack.py"; then
    echo
    echo "STOP: unless the message above says NO GPU, fix the venv before running anything"
    echo "      (gpumode_local/reference/README.md)."
    exit 1
fi

echo
echo "== compiling and running a real Triton kernel on this card =="
# In a file, not on stdin: triton.jit calls inspect.getsourcelines on the kernel, which
# fails with "could not get source code" for anything fed through a heredoc.
cat > "$TMP/smoke.py" <<'SMOKE_EOF'
import sys
import torch, triton, triton.language as tl


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


BUSY = ("BUSY  the card has no free memory -- something else (vLLM?) owns it.\n"
        "      This says nothing about the toolchain. Point CUDA_VISIBLE_DEVICES at a\n"
        "      free card and re-run; inside an LSF job the scheduler sets it for you.")

try:
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    _add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    correct = torch.allclose(out, x + y)      # inside the try: this allocates too
except torch.cuda.OutOfMemoryError:
    sys.exit(BUSY)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        sys.exit(BUSY)
    raise

if correct:
    print("PASS  triton compiled and ran correctly on this card")
else:
    sys.exit("FAIL  triton ran but produced wrong results")
SMOKE_EOF

if "$KPY" "$TMP/smoke.py"; then
    echo
    echo "Grading stack is good. Record the reference baseline next (README step 3)."
else
    echo
    echo "STOP: unless the message above says BUSY, triton cannot compile for this card."
    echo "      On a B200 that is the cu126-vs-cu128 mistake -- look for ptxas errors"
    echo "      mentioning 'unsupported' or 'target'."
    exit 1
fi
