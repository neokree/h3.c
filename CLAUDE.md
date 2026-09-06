@AGENTS.md

## Language

Talk to the owner in **Italian**. Everything written down stays in **English**:
code, comments, commit messages, issues, issue comments, and documentation.

## Agent skills

### Issue tracker

GitHub Issues on **`neokree/h3.c`** (a fork; `gh` needs `-R` or a set-default, see the doc).
See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
The glossary is loaded into every session, so use its terms rather than synonyms.

`docs/lora.md` is the runtime-LoRA mechanism: what the shipped code does, and the
measurements behind each decision. Read it before changing anything under
`h3_lora.*`, the delta branch in `h3_dit.c`, or the memory guardrail.

@CONTEXT.md

### Wayfinder map

The runtime-LoRA design map is issue #1: https://github.com/neokree/h3.c/issues/1
