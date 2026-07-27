# ICL runs — copy-paste command log. Options are documented in ../README.md.


# ============ Start the model (vLLM server) ============

ATTENTION: max-model-len can be adjusted depending on n-context. Rough rule: 


INESC ID machines:

cd projects/phd/R2/LLMs/local/vllm_provider/
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/scratch/vicstorage \
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --max-model-len 130000 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8
    --reasoning-parser openai_gptoss \
    --port 8001


---------------------

Bosch machines:

source ~/envs/bin/activate
vllm serve Qwen/Qwen3.6-27B-FP8 \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --max-model-len 130000 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --reasoning-parser openai_gptoss \
    --kv-cache-dtype fp8
    --port 8001
    


# ============ ICL runs ============
cd projects/phd/learning_evolve/src/
source .venv/bin/activate

# circle_packing_26 — context-strategy sweep (g5x12, gen30)

python run_icl.py --problem circle_packing_26   --groups-per-batch 5 --group-size 12 --num-generations 30     --n-context 0    --reasoning-effort medium --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --log-path runs/cp_26_no_icl_g5x12_gen30

python run_icl.py --problem circle_packing_26   --groups-per-batch 5 --group-size 12 --num-generations 30     --n-context 30 --context-strategy random    --reasoning-effort medium --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --log-path runs/cp_26_random_n_30_g5x12_gen30

python run_icl.py --problem circle_packing_26   --groups-per-batch 5 --group-size 12 --num-generations 30     --n-context 30 --context-strategy best     --reasoning-effort medium --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --log-path runs/cp_26_best_n_30_g5x12_gen30

python run_icl.py --problem circle_packing_26   --groups-per-batch 5 --group-size 12 --num-generations 30     --n-context 30 --context-strategy best_worst --mix-fraction 0.7     --reasoning-effort medium --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --log-path runs/cp_26_best_worst_n_30_g5x12_gen30

python run_icl.py --problem circle_packing_26   --groups-per-batch 5 --group-size 12 --num-generations 30     --n-context 30 --context-strategy contrastive --mix-fraction 0.7 --mmr_lambda 0.7    --reasoning-effort medium --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --log-path runs/cp_26_contrastive_n_30_g5x12_gen30
