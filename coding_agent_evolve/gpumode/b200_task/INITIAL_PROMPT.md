# TriMul on a B200 — agent run

<!-- Zone 1 below is `TRIMUL_PROMPT` from src/envs/kernel_prompt.py, verbatim, so this arm
     and the ICL arm are handed the same task. The hardware line is `_HW_RULE_B200` from
     src/envs/kernel_trimul.py, verbatim. Everything after "## How you work" is
     agent-specific and has no ICL counterpart -- that difference IS the experiment. -->

You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C]. 

Your task:
- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:
Input:
- `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    - input: Input tensor of shape [bs, seq_len, seq_len, dim]
    - mask: Mask tensor of shape [bs, seq_len, seq_len]
    - weights: Dictionary containing model weights
    - config: Dictionary containing model configuration parameters

Output:
- output: Processed tensor [bs, seq_len, seq_len, dim]

**Problem Constraints:**
- B ∈ {1,2}, N ∈ {128,256,512,1024}, c ∈ {128}, c_z ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

**Remarks.** So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N. 
The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!). 
However, I really like this kernel because it only consists of “simple” operations, and is really easy to understand. It is a true test of “fusions” that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)
        
        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data)
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul

    return out
```

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:
- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):
  - {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}

## Rules

- The tensors arguments passed in will be already on your cuda device.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an NVIDIA B200 (sm100, Blackwell).
- The B200 allows at most 232448 bytes of shared memory per kernel launch, so block sizes or
  num_stages needing more than that will fail to launch. It has fp8 tensor cores.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.

## Scoring

Your kernel is scored by the frozen harness in this folder. **Do not modify `evaluate.py`
or anything under `trimul/`** — if the harness rejects a candidate, the candidate is at
fault until you prove otherwise.

```bash
$KPY evaluate.py candidate.py --mode test          # 18 correctness shapes only, ~30 s
$KPY evaluate.py candidate.py --mode leaderboard   # THE score: correctness, then 7 timed shapes
$KPY evaluate.py candidate.py --mode leaderboard --repeats 5 --json out.json
```

`$KPY` is set for you and names the only interpreter with a torch/triton build that has
kernels for this card. Never run `./evaluate.py` or a bare `python evaluate.py`: the
harness only produces meaningful timings against that exact stack, and a different
interpreter either fails confusingly or, worse, succeeds and lies.

**Never pass `--gpu`.** The grader inherits `CUDA_VISIBLE_DEVICES`, which the scheduler
sets to the card allocated to this job. Naming an index overrides that and can point your
timings at another job's GPU.

The score printed as `SCORE (geom of 7 benchmarks): <us>` is the geometric mean runtime in
microseconds across the benchmark shapes. **Lower is better.** Exit code is 0 only if
everything passed. `leaderboard` is the ranked path and the only mode whose numbers count;
`benchmark` skips the correctness gate and lands a couple of percent elsewhere, so never
compare a benchmark number to a leaderboard one.

## Your objective

There is no seed kernel and no target time. You start from nothing but the PyTorch
reference above, and the objective is to drive the leaderboard score as low as you can.

Your first task is to establish a baseline: write a working kernel, score it in
`leaderboard` mode, and record it as the champion. Everything after that is measured as
verified improvement over the current champion.

## How you work

- **Maintain `best.py`** — the current champion. Replace it only with a candidate that
  passed the full correctness suite AND beat the champion's *freshly measured*
  leaderboard score in the same session. Never edit it in place.
- **Maintain `LEDGER.md`** — one row per candidate: approach family, one-line description,
  correctness pass/fail, leaderboard score, delta vs champion, verdict. Failed candidates
  are search signal, not waste: record the concrete failure cause (wrong results / too
  slow / compile error / out of shared memory) and keep the file in `attempts/`.
- **Group by idea, not by wording.** Maintain the ledger's approach families explicitly.
  If several candidates converge on one family, deliberately push some effort into an
  underexplored formulation: different fusion boundaries, different data layouts,
  different decompositions of the N³ contraction, different numerical strategies,
  delegating more or less to PyTorch.
- **Measure, do not assume.** Any claim about why something is fast or slow needs a
  number behind it. An unmeasured kernel is not progress.
- **Beware measurement noise.** Run-to-run spread is around 1 % on an idle card. A win
  inside the noise is not a win — use `--repeats` before believing a small one. Never
  compare against a stale champion number; re-measure both back to back.
- **When a route stalls**, mark it BLOCKED in the ledger with the evidence that blocked it
  (a profile, a resource limit, repeated verified regressions). Reopen it only if you have
  a genuinely new mechanism that addresses that evidence.
- Triton 3.3.1's Blackwell support is first-generation: it compiles for sm100 but exposes
  none of the newer machinery (tcgen05 MMA, the newer TMA descriptors). If you see ptxas
  errors mentioning `unsupported` or `target`, that is the toolchain, not your kernel.

Return when you cannot improve the champion further. Report the champion's score, the
approach families you tried, and what blocked the ones that failed.

## No web access

You have **no web access**. Do not attempt to search the web or fetch a URL — the tools
are disabled and every attempt is a wasted turn. Everything must come from your own
reasoning and from experiments you run on this machine.
