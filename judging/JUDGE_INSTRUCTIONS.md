# Estimate-judge instructions (for judge subagents)

You are acting as a mechanical LLM judge for a research pipeline. Your input file (given in your task prompt) is JSONL; each line has fields "prompt_hash" and "rendered".

The "rendered" field is a complete, self-contained judging instruction: extract the final single-number estimate from a model's text into `<final_estimate>...</final_estimate>` tags; output UNKNOWN for ranges / uncommitted / ambiguous answers; convert shorthand ("2.5 million" → 2500000). Some prompts instead ask for a duration converted to whole days — follow whichever rules the rendered text states.

For EACH record independently, follow its rendered instruction EXACTLY and strictly — when in doubt per its rules, output UNKNOWN. Do not let one record influence another.

Write your results to the output file given in your task prompt: one JSON line per input record, same order, each of the form:

    {"prompt_hash": "<copied verbatim from input>", "answer": "<final_estimate>NUMBER_OR_UNKNOWN</final_estimate>"}

The "answer" value must contain ONLY the final_estimate tag. Process every record (100 per chunk). You may use python via Bash to read/write files, but the judgement of each text must come from you reading it. In your final message report only: count processed + output path.
