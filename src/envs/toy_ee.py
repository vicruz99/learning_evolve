"""Synthetic smoke-test problem ("toy").

This problem is **not** a real discovery task. It exists only to validate that the ICL harness is
wired correctly end-to-end (prompt respected, ``<strategy>`` parsed, PUCT buffer fills, context block
filled with past solutions + scores, artifacts/tracking written) and to eyeball whether the best
solutions visibly *converge* toward something.

The model is told it is a fake test and is asked for one short, human-readable sentence. It is
**never** told how it is scored. The hidden metric is simply the number of ``"ee"`` substrings in the
sentence (case-insensitive), so a model that infers the pattern from the scored examples in its
context should drift toward sentences containing more "ee".

Unlike the real problems, grading is **in-process** (no Ray sandbox): the model emits the sentence
directly inside a ```` ```text ```` fence, which the harness de-fences (see the ``_get_code_languages``
/ ``_should_keep_code_separators`` overrides below) and hands to :meth:`ToyEeReward.get_reward` as a
bare string. ``ToyEeEnv.uses_sandbox = False`` tells the loop to skip Ray initialization entirely.
"""
from __future__ import annotations

from envs.base import Environment
from puct import State
from sandbox.base_reward_evaluator import BaseRewardEvaluator


def _clean_sentence(code: str) -> str:
    """Bare sentence from the parsed block.

    Defensive: if the model puts the closing ``` on the same line as the sentence, the shared
    codeblock parser can leave a trailing fence in the content (it only strips a fence preceded by a
    newline). Drop any trailing ``` so the stored sentence and its score stay clean.
    """
    s = (code or "").strip()
    while s.endswith("```"):
        s = s[:-3].rstrip()
    return s


class ToyEeReward(BaseRewardEvaluator):
    """In-process evaluator (hidden from the model).

    Score = "ee"-density = count of ``"ee"`` divided by the sentence length in characters. Dividing
    by length rewards sentences that are *dense* in "ee" rather than merely long, so convergence
    should push toward short strings packed with "ee".
    """

    def __init__(self, problem_type, log_dir, eval_timeout: int = 30, num_cpus_per_task: int = 1, **kwargs):
        # Accept the four kwargs the harness passes (base.Environment._run_verification); no Ray.
        self.problem_type = problem_type
        self.log_dir = log_dir

    def get_reward(self, code: str, state: State) -> dict:
        # ``code`` is the already de-fenced sentence (see env overrides below).
        sentence = _clean_sentence(code)
        if not sentence:
            return {"reward": 0.0, "correctness": 0.0, "raw_score": 0.0, "msg": "empty sentence"}
        n_ee = sentence.lower().count("ee")            # hidden metric; case-insensitive
        density = n_ee / len(sentence)                 # "ee" per character
        return {
            "reward": density,
            "correctness": 1.0,
            "raw_score": density,
            "msg": f"len={len(sentence)} ee={n_ee} density={density:.4f} :: {sentence[:120]}",
            "result_construction": [],  # do not carry a construction across states
            "stdout": "",
        }


class ToyEeEnv(Environment):
    reward_function = ToyEeReward
    state_type = State
    uses_sandbox = False  # in-process grading; the loop skips init_ray for this problem

    def is_maximize(self) -> bool:
        return True  # higher score is better

    def _get_code_languages(self) -> list[str]:
        # The model wraps its sentence in a ```text fence; extract that block.
        return ["text"]

    def _should_keep_code_separators(self) -> bool:
        # Hand get_reward the bare sentence (no ```text ... ``` wrapper).
        return False

    def get_question(self) -> str:
        # The parent PUCT selected to expand. Show it (as the real envs do via State.to_prompt) so
        # the model actually sees the specific sentence it is meant to improve on. At gen 0 the parent
        # is the empty seed, so we ask for a fresh sentence instead.
        st = self.initial_state
        parent_sentence = _clean_sentence(getattr(st, "code", ""))
        if parent_sentence:
            score = st.value if self.is_maximize() else -st.value
            current_block = (
                f"You are refining an existing sentence. Here is the current sentence "
                f"(score = {score:.4f}):\n```text\n{parent_sentence}\n```\n"
                f"Produce a NEW single sentence that scores higher than this one.\n\n"
            )
        else:
            current_block = "There is no starting sentence yet — write a fresh one.\n\n"

        return f"""You are helping to test a software pipeline. This is a SYNTHETIC test task, not a \
real problem, so do not overthink it and do not spend long reasoning about it.

Your job: write ONE short, ordinary sentence about anything at all (a fact, an observation, a bit of \
fiction — your choice). Keep it to a single sentence.

An automatic scorer will assign your sentence a numeric score. You are NOT told how the score is \
computed. Higher scores are better. {current_block}If past attempts are shown below, look at their \
scores and try to produce a sentence that scores as high as possible.

Output format (follow it exactly):
1. First, put a brief note about your approach between <strategy> and </strategy> tags.
2. Then give your sentence, and nothing else, inside a fenced block marked ```text like this:

```text
Your single sentence goes here.
```
"""
