# Choice-judge instructions (for judge subagents)

Your input file (named in your task prompt) is JSONL; each line has "rid" and "rendered". The "rendered" field is a complete, self-contained judging instruction: it shows a JSON case (two activities + an assistant response) and asks you to classify what the assistant did, returning `{"choice": 1}`, `{"choice": 2}`, or `{"choice": "refusal"}` under precisely stated rules — read those rules carefully; the two "crucial distinctions" (capability disclaimers are NOT refusals; eliminating one option and defaulting to the other IS a refusal) matter often.

For EACH record independently, follow the rendered instruction exactly. Write one JSON line per input record, same order, to the output file named in your task prompt:

    {"rid": "<verbatim>", "choice": 1 | 2 | "refusal"}

Process every record. ISOLATION: intermediates ONLY inside the unique work dir named in your task prompt (create it) — never the shared scratchpad. Final message: count + tally of 1/2/refusal + output path.
