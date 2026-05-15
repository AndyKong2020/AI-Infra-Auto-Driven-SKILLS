---
name: model-architecture-ascii
description: Canonical structural prior for in-scope LLM / MoE / VLM families — block-level Mermaid diagram, per-block module decomposition with closed-vocabulary type tags, model-level metadata, and parameter tables, in a single agent-readable Markdown file per model. Use as the starting point whenever an agent task involves model architecture decomposition, profiling analysis, kernel optimization, perf attribution by module/layer, or any similar work where on-the-spot derivation would risk inconsistent decomposition granularity or correctness across agents. Stock HF model only — quantization / parallelism / kernel-variant information is out of scope and must be overlaid by the consumer. Coverage is opt-in per-model: only models with a file under references/diagrams/ are returned.
---

# Model Architecture (Mermaid)

> **Status: MVP style-check.** Resolver script is intentionally not implemented yet; current goal is to validate format + style + policy via 3 representative diagrams. Once locked, the resolver lands on top of the agreed format.

## Purpose

When an agent in this repo is doing **model architecture decomposition**, **profiling analysis**, **kernel-level optimization**, **perf attribution**, or similar work, it should first read the corresponding model family's structural prior from this skill, then analyse / fine-tune on top of that. The prior collapses two failure modes of on-the-spot derivation:

- **Correctness drift** — different agents get different forward graphs for the same model (e.g. missing DeepSeek MoE's shared expert, mistaking MLA for vanilla MHA).
- **Granularity drift** — different agents decompose the same block into different numbers of modules; profiling baselines across sessions don't align.

This skill is the single source of truth for both. Same prior every session.

## What each entry contains

For each model with a file under `references/diagrams/<id>.md`:

1. **Frontmatter** — id, title, aliases, rank, three model-level closed-vocabulary tags (`modality`, `attention_type`, `ffn_type`), source_basis.
2. **`## Model summary`** — model-level orthogonal tags (modality, attention_type, ffn_type) + scale numbers (total_params, active_params). Block-level diagrams only. Deliberately omits deployment-specific fields like `context_length` and `precision`.
3. **Mermaid `flowchart TD` block** — visual structure; source-text is the agent-parseable form of the graph.
4. **`## Modules (in forward order)`** — closed-vocabulary `type` tag per module + `Count` column = number of times each module is instantiated in the full forward pass. Block-level diagrams only.
5. **`## Key parameters`** — detailed numerical fields (lora ranks, head dims, intermediate sizes, ...) from `config.json`.
6. **`## Notes`** — quirks not captured by the graph: dense-vs-MoE layer ranges, KV cache shape, routing rules, ...
7. **`## Source basis`** — pointers to the reference image (if any), `config.json`, and `modeling_*.py`.

Module-detail diagrams (e.g. `deepseek-v3.2-mla.md`) omit `## Model summary` and `## Modules` (the whole file *is* the inside of one module).

## Scope

- **In scope**: LLM, MoE-LLM, VLM (text + vision), hybrid-attention LLM. See `references/authoring-policy.md` § 4 for the model list.
- **Out of scope**: diffusion / video / 3D generation models — these stay with the sibling image-based `model-architecture-diagram` skill.
- **What this skill does NOT cover**: deployment-specific information (quantization configs, parallelism shard layouts, optimized kernel variants, runtime fusions). Downstream agents must overlay these on top of the stock prior given here.

## Reading order for new authors / agents

1. `references/authoring-policy.md` — what to draw, how many diagrams per model, closed-vocabulary tags for `type` and `family_type`.
2. `references/mermaid-style-guide.md` — how to draw each diagram (shapes, label rules, fixed section order).
3. `references/diagrams/*.md` — existing entries to use as templates.

## Consumer guarantees

- `modality`, `attention_type`, `ffn_type` (model-level) and `type` (per-module) come from closed vocabularies declared in `authoring-policy.md`; downstream agents can rely on stable tag values for filtering / classification along independent axes.
- `Modules` table row order = forward-pass execution order. Profiling agents can fold `Count` × per-invocation cost without re-deriving order.
- Mermaid block source is canonical: node ids and labels are stable across edits unless a structural change is intended.
