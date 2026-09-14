"""Hold sampling equal between the Claude Code arm and the bnbcode arm.

The bnbcode arm sets `agent.build.temperature = 0.6` / `top_p = 0.95` in opencode.json.
Claude Code has no equivalent knob: it sends its own temperature on every request, and a
value in `litellm_params` is only a default that the request overrides. So the two arms
would silently run at different sampling settings -- a difference that is not the harness,
and would contaminate the comparison the experiment exists to make.

This is a LiteLLM proxy callback that overwrites both fields after the request is parsed
and before it is sent upstream. Wired in litellm_claude.yaml:

    litellm_settings:
      callbacks: cc_sampling_hook.handler

It logs the first request it touches, including what the client had asked for, so the
override is visible in the proxy log rather than taken on faith:

    grep cc-sampling-hook ~/agent_runs/litellm.<host>.log
"""
from __future__ import annotations

try:                                              # litellm >= 1.9x
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:                               # pragma: no cover - older layouts
    from litellm.integrations.custom_logger import CustomLogger  # type: ignore

TEMPERATURE = 0.6
TOP_P = 0.95
REASONING_BY_ALIAS = {
    "low": {"reasoning_effort": "low"},
    "medium": {"reasoning_effort": "medium"},
    "xhigh": {"reasoning_effort": "xhigh"},
    "off": {"enable_thinking": False},
}


class SamplingParity(CustomLogger):
    """Pin temperature and top_p on every completion the proxy forwards."""

    def __init__(self) -> None:
        super().__init__()
        self._announced = False

    async def async_pre_call_hook(self, user_api_key_dict=None, cache=None, data=None,
                                  call_type=None, **kwargs):
        # LiteLLM passes these as keyword arguments and accepts a dict (or None) back.
        # Signatures have gained arguments across releases, hence **kwargs: an unexpected
        # one must not take the proxy down mid-run.
        if not isinstance(data, dict):
            return data
        asked = (data.get("temperature"), data.get("top_p"))
        data["temperature"] = TEMPERATURE
        data["top_p"] = TOP_P
        # v2: reasoning effort travels in the model alias (qwen3.8-<low|medium|xhigh|off>).
        # litellm_qwen38.yaml already puts the matching chat_template_kwargs in extra_body;
        # this is the belt to that brace, for a LiteLLM release that stops forwarding it.
        alias = str(data.get("model", ""))
        if "chat_template_kwargs" not in data and alias.startswith("qwen3.8-"):
            kw = REASONING_BY_ALIAS.get(alias.rsplit("-", 1)[-1])
            if kw:
                data["chat_template_kwargs"] = dict(kw)
        if not self._announced:
            print(f"[cc-sampling-hook] pinning temperature={TEMPERATURE} top_p={TOP_P} "
                  f"on every request (client asked temperature={asked[0]!r} "
                  f"top_p={asked[1]!r}, call_type={call_type!r})", flush=True)
            self._announced = True
        return data


handler = SamplingParity()
