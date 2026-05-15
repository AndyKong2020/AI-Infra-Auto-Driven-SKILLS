---
name: model-architecture-ascii
description: Return Mermaid-in-Markdown renderings of model architectures maintained in this skill's references/. Use when the user asks for a text-source / agent-friendly architecture sketch, when the output will be consumed by another LLM, or as a complement to the image-based model-architecture-diagram skill. Coverage is opt-in per-model — only models with a file under references/diagrams/ are returned.
---

# Model Architecture (Mermaid)

> **Status: MVP style-check.** Resolver script and authoring playbook are intentionally not implemented yet. Current goal is to validate Mermaid-in-Markdown as the output format using a small set of representative diagrams. Once style is confirmed, this SKILL.md will be filled in with the workflow, CLI interface, and authoring playbook agreed in the design discussion.

## What this skill stores

Each entry under `references/diagrams/<id>.md` contains:

1. YAML frontmatter (id, title, aliases, rank, source_basis)
2. A `flowchart TD` Mermaid block — the visual diagram
3. A `## Key parameters` table — config-derived numerical fields
4. A `## Notes` section — quirks not visible in the graph
5. A `## Source basis` section — pointer to the original image (if any), config.json, and modeling code that the entry was transcribed / cross-checked against

Cross-references between entries use wiki-style `[[other-id]]` links.

## Why Mermaid-in-Markdown (not raw ASCII)

- **Renders in any modern Markdown viewer** (GitHub web, Claude.ai, Cursor, VS Code) — pretty visual output without external image URLs.
- **Source is agent-native**: another Claude session reading the `.md` file extracts topology from declarative Mermaid, parameters from the table, and quirks from the Notes — no char-art parsing required.
- **Terminal `cat` fallback still useful**: even without a Mermaid renderer, the param table and Notes are fully readable; only the visual diagram degrades to source.

See `references/mermaid-style-guide.md` for diagram conventions.

## MVP sample set

The diagrams currently committed are picked to stress-test the style guide on three different topology classes:

- `deepseek-v3.2-architecture` — full-model block view with `subgraph` for `Block × N` and named MoE/MLA modules.
- `deepseek-v3.2-mla` — module detail with multi-branch fan-out / RoPE injection / concat.
- `qwen3-moe-block` — second model family (different MoE flavor: no shared expert) to verify the style guide isn't DeepSeek-specific.

## Authoring policy

Before adding a new model, read `references/authoring-policy.md`. It defines:

- the **sourcing rule** (anchor topology to the reference image, take numbers from `config.json`, verify graph with `modeling_*.py`, refine editorially only when noted);
- the **module catalogue** that decides how many diagrams a model gets (which modules auto-earn a detail diagram, which never do, which are conditional);
- the **in-scope model set** (LLM / MoE / VLM in scope; diffusion / video / 3D out of scope for this skill).

## References

- `references/authoring-policy.md` — what to draw, how many, and from which sources.
- `references/mermaid-style-guide.md` — how to draw each diagram (shapes, label rules, file structure).
- `references/diagrams/*.md` — diagram entries, one per artifact.
