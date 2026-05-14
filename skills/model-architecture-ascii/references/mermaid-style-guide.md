# Mermaid style guide

Conventions for diagrams stored in `references/diagrams/*.md`. Applies to all new entries.

## Direction

- Always `flowchart TD` (top-to-bottom). Transformer architectures are layer stacks, so vertical flow matches the mental model.
- Inside a `subgraph`, you may set `direction LR` only when the inner content is *naturally* a horizontal fan-out (e.g. a router dispatching to parallel experts). Default inside a subgraph is also `TB`.

## Abstraction ceiling

- Match the level of detail shown in the original architecture image (if there is one) — see the sibling `model-architecture-diagram` skill for reference.
- Do **not** drill below module level: no GEMM decomposition, no element-wise op breakdown, no kernel/tile-level steps.
- If a module deserves more detail than fits at the top-level, create a separate `.md` file (e.g. `deepseek-v3.2-mla.md`) and link to it from the top-level diagram via `[[id]]` and a Notes line.

## Node conventions

| Role | Shape | Mermaid syntax |
|---|---|---|
| Input / output tensor | rounded capsule | `id([label])` |
| Module / op / projection | rectangle | `id[label]` |
| Wide-but-named module (MLA Attn, MoE FFN, GQA Attn) | rectangle with `<br/>` for secondary detail | `id["MLA Attn<br/>q_lora=1536, kv_lora=512"]` — keep secondary detail short, push numbers into the Notes table when in doubt |
| Join / concat | parenthesised or double-circle | `id((·))` or `id([concat per-head])` |

Avoid stadium / hexagon / trapezoid shapes — too many shapes hurts uniformity across the skill.

## Edges

- Default arrow `-->`.
- Residual sum: `-->|+ residual|` inline label. Use sparingly — one per residual stream is enough; don't tag every arrow.
- Cross-head broadcast / shared tensor: label the edge `-->|shared|` or note in `## Notes`.

## Repeated blocks

Repeated layer stacks (`Block × N`) live inside a `subgraph` with explicit count in the label:

```
subgraph block ["Block × 61"]
    direction TB
    n1[RMSNorm]
    attn[MLA Attn]
    ...
end
```

External edges enter the first inner node and exit from the last:

```
embed --> n1
moe -->|+ residual| fnorm
```

Don't draw `Block × N` as a single rectangle with `× N` in the label — losing the inner topology defeats the point of the diagram.

## Sections in each entry file

Fixed order, every entry:

1. Frontmatter (id, title, aliases, rank, source_basis)
2. Top-level heading `# <Title>`
3. Mermaid `flowchart TD` block — the diagram itself, no narration before it
4. `## Key parameters` table — only fields that affect inference shape or runtime behavior (n_layers, hidden, vocab, n_heads, lora ranks, n_experts, top_k, E_inter). Skip ones that don't.
5. `## Notes` — quirks not visible in the graph: dense-vs-MoE layer ranges, KV cache shape, RoPE specifics, special routing rules, quantization points, cross-refs via `[[id]]`.
6. `## Source basis` — what the entry was transcribed / verified against.

## Naming

- `id`: kebab-case slug, matches filename without `.md`. Examples: `deepseek-v3.2-architecture`, `deepseek-v3.2-mla`, `qwen3-moe-block`.
- `title`: human-readable, include the model name and what level the diagram is at. Examples: `DeepSeek V3.2 architecture (block-level)`, `DeepSeek V3.2 MLA forward`, `Qwen3 MoE block`.
- `aliases`: include HF id, colloquial names, common variants. If two models share architecture (DeepSeek V3 ↔ R1), share aliases on the same entry.

## Cross-references

- Wiki-style `[[other-id]]` inside Notes or Source basis.
- The resolver will **not** auto-expand `[[id]]`; the reader (human or agent) re-invokes the skill if they want the linked diagram.

## Width budget

Mermaid auto-layouts, but keep node labels under ~30 characters. Long parameter lists belong in the `## Key parameters` table, not in node labels.
