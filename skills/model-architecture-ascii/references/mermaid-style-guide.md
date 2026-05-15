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

Three shapes, each with one role. No overlap.

| Role | Shape | Mermaid syntax |
|---|---|---|
| Flow endpoint (input / output of the diagram, e.g. `input tokens`, `logits`, `x`, `output`) | capsule (stadium) | `id([label])` |
| Module / op / projection / tensor | rectangle | `id[label]` |
| Merge / join / concat / fan-in | circle | `id((label))` |

Avoid the other Mermaid shapes (subroutine `[[ ]]`, cylinder `[( )]`, diamond `{ }`, hexagon `{{ }}`, parallelogram `[/ /]`, double-circle `((( )))`). Each extra shape forces the reader to decode another visual convention; sticking to three keeps the skill uniform across hundreds of future entries.

### Shape gallery

The three shapes side by side (renders on any Mermaid-aware viewer):

```mermaid
flowchart LR
    a([capsule · flow endpoint])
    b[rectangle · module or tensor]
    c((circle · merge))
    a --> b --> c
```

Use this as the canonical reference when picking a shape for a new node.

## Node label content (hard rules)

These rules keep module names and annotations visually distinct, and keep numerical params in a single source of truth.

**Module nodes** (modules / ops / projections in the data flow):

- **Name only.** No numerical parameters in the label. Example: `attn[MLA Attn]`, `moe[MoE FFN]`, `n1[RMSNorm]`.
- All numerical params (`n_heads`, `q_lora_rank`, `n_routed_experts`, ...) go into the `## Key parameters` table, not into node labels.
- **Allowed exception — distinguishing-feature italic line.** When a module has a feature that differs from a same-named module in another well-known model (e.g. Qwen3 MoE has no shared expert vs DeepSeek MoE has 1), you may add a single italic line via HTML:

  ```
  moe["MoE FFN<br/><i>no shared expert</i>"]
  ```

  Use this only for *distinguishing* facts, not for parameter values. One italic line max per node.

**Tensor / data-flow nodes** (latent tensors named in the forward pass, e.g. `c_Q`, `q_nope`, intermediate activations):

- Format `name : shape` with `:` as separator (do not use `·` or `dim`).
- Examples:
  - `cQ["c_Q : 1536"]`
  - `q_nope["q_nope : 128h × 128"]`
  - `embed["Embed : V → 7168"]` (here shape arrow encodes the projection)
- Shape is part of the data-flow story, so it earns its place inside the label.

**Cross-cutting facts** (things true about the model but not localized to one node — "layers 1–3 use dense FFN", "KV cache stores c_KV+k_rope", "first 4 layers don't apply RoPE"):

- Always go into `## Notes`, never into node labels.

**Edge labels** (text on an arrow):

- Reserved for short structural annotations on the flow itself: `-->|+ residual|`, `-->|shared|`, `-->|top-8|`. Don't use edge labels for module parameters.

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
