"""The pinned out-of-band ("subagent") judge identity.

This dict is hashed into every estimate-judge cache path
(`judge_config_hash` uses model / max_tokens / temperature /
reasoning_effort). NEVER change these values once caches exist, or every
judged rollout forks into a fresh unreachable cache directory.

The `manual` backend never calls an API: on a cache miss
`batch_extract_estimates` dumps the rendered judge prompts to
`$MANUAL_JUDGE_PENDING_PATH` and raises `ManualJudgePending`. The prompts
are then judged out-of-band (Claude Code subagents) and written back via
`local_tools.write_judgements`.
"""

SUBAGENT_JUDGE_CONFIG = {
    "backend": "manual",
    "model": "claude-subagent",
    "max_tokens": 1024,
    "temperature": 0,
    "max_concurrent": 1,
}
