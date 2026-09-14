venv for experiments:
/home/crv1pi/venvs/agent-eval/bin/python

The runs are kept here: (but theres a symlink for home folder, the agent must be launched
from this one, not from the symlink)

~/agent_runs/<problem>_<variant>_s<seed>          bnbcode arm
~/agent_runs/<problem>_<variant>_cc_s<seed>       Claude Code arm

    tmux new -s cpu1
    bsub -Is -q batch_cpu -J cpu1 -P BH-000557-01 -n 32,128 -W 110:00 -M 4096 \
         -R "rusage[mem=4096]" -R "span[hosts=1]" /bin/bash

    cd /home/crv1pi/work/learning_evolve/coding_agent_evolve/experiments

To get everything ready to run, source the launcher for the arm you want. Both are safe to
re-run, both start only what is missing, and both must be SOURCED, not executed:

    source ~/bin/agent-up          # bnbcode arm
    source ~/bin/claude-up         # Claude Code arm

Each one prints the exact copy-paste recipe for its arm when it is done. They share the
same relay and the same vLLM, so running both on one node is fine.

If `bnbcode: command not found`: it lives in ~/work/bnbcode/.venv/bin and is symlinked into
~/.local/bin, which agent-up puts on PATH. Re-source agent-up. Same for `claude`.

Details, and the parity table for the two arms: OPERATOR_NOTES.md sections 5, 6 and 9.
