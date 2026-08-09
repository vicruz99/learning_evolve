# Claude Code Guidelines

## Critical Context & Reading List
At the beggining of each session, you MUST read these files to get all the important info releated to the project:
- `docs/PROJECT_CONTEXT.md` (Core project rules, architecture, and current goals)
- `docs/EXPERIMENT_PLAN.md` (Notes on current experiments, and experiments in general)
- `docs/IMPLEMENTATION_LOG.md` (Dated diary of what was built, design decisions, and problems faced/solved)

Read when the work touches the Bosch cluster (submitting jobs, queues, eval slowness, timeouts):
- `docs/BOSCH_CLUSTER.md` (LSF queues/hosts, imposed defaults, and why the login node must never be used for compute)

Read when the work touches launching or restarting experiments:
- `docs/SWEEPS.md` (`run_sweep.py`: launch/status/stop, `--resume` vs `--continue-run`, the Ray flags)

## Project Structure & Workspace Warning
- **Nested Repositories:** This project folder will contain/contains two cloned external repositories (ShinkaEvolve and discover (repo for TTT-Discover))
- Do NOT accidentally modify files inside these cloned sub-repositories unless explicitly instructed to do so.
- Keep all primary development focused on the /src root project files.