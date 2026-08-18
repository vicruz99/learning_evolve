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
"$KPY" - <<'PY'
import sys
import torch, triton

props = torch.cuda.get_device_properties(0)
cc = torch.cuda.get_device_capability(0)
sm = f"sm_{cc[0]}{cc[1]}"
archs = torch.cuda.get_arch_list()

print(f"torch            {torch.__version__}")
print(f"triton           {triton.__version__}")
print(f"card             {props.name}")
print(f"capability       {sm}")
print(f"arch list        {archs}")
print(f"shared mem/block {props.shared_memory_per_block_optin}   <- the number the B200 prompt quotes")

ok = True
if triton.__version__ != "3.3.1":
    print(f"\nFAIL  triton is {triton.__version__}, the harness pins 3.3.1"); ok = False
if not torch.__version__.startswith("2.7.1"):
    print(f"\nFAIL  torch is {torch.__version__}, the harness pins 2.7.1"); ok = False
if sm not in archs:
    print(f"\nFAIL  this interpreter has no kernels for {sm}. On a B200 you need the cu128"
          f"\n      build -- sm_100 gencode only comes from CUDA 12.8."); ok = False
if sm == "sm_100" and "cu128" not in torch.__version__:
    print(f"\nFAIL  {sm} needs a cu128 wheel; this is {torch.__version__}"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo; echo "STOP: fix the venv before running anything (gpumode_local/reference/README.md)"; exit 1; }

echo
echo "== compiling and running a real Triton kernel on this card =="
"$KPY" - <<'PY'
import sys
import torch, triton, triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 4096
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
_add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
if torch.allclose(out, x + y):
    print("PASS  triton compiled and ran correctly on this card")
else:
    print("FAIL  triton ran but produced wrong results"); sys.exit(1)
PY
[ $? -eq 0 ] && echo && echo "Grading stack is good. Record the reference baseline next (README step 3)."
