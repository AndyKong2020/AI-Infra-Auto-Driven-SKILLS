# Authoring policy

How to decide **what to draw**, **how many diagrams per model**, and **which closed-vocabulary tags to apply**, when adding an entry to this skill. Companion to `mermaid-style-guide.md` (which covers *how to draw*).

Read this before adding any new model.

## 0. Primary consumer

The primary consumer of this skill's output is **another agent** doing model architecture decomposition, profiling, kernel optimization, or similar work. Visual rendering for humans is a side benefit.

This affects every other rule in this document:

- Formats must be **agent-parseable** (Mermaid source, tables with closed-vocabulary tags, prose Notes).
- Decomposition granularity must be **deterministic** (same model → same module list across sessions).
- Tag vocabularies must be **closed and stable** (no silent invention; extend the policy first).
- Scope is **stock HF model only**: deployment overlays (quantization, parallelism, kernel choice) are explicitly out of scope and left to the downstream consumer.

## 1. Sourcing policy: "anchor + refine"

For each diagram, three sources feed in, with clear roles:

| Source | What it provides | Authority |
|---|---|---|
| Reference image in the sibling `model-architecture-diagram` skill (if one exists for this model) | Topology skeleton, module naming, abstraction level | Community alignment — use this so readers recognise the diagram |
| HF `config.json` for the model | All numerical fields (n_layers, hidden, vocab, n_heads, lora ranks, n_experts, top_k, etc.) | Ground truth for numbers |
| `modeling_*.py` (HF transformers code) | Block composition verification, forward-pass details, residual paths | Ground truth for graph composition |
| Editorial judgment | Targeted refinements only when the reference image is ambiguous in a way that hurts the reader | Last resort — must be noted in `## Source basis` |

**Resolution rule**: when the image and the code disagree, the code wins for graph composition; the image wins for naming/abstraction level. Numerical values always come from `config.json`, never from the image (the image usually has symbolic placeholders).

When the original image leaves a path implicit but the path is important for an implementer (kernel writer, KV-cache planner, etc.), make it explicit and record the deviation in `## Source basis`. Example: the DeepSeek MLA reference image keeps the `W_KR → k_rope` branch implicit; our MLA detail diagram draws it explicitly.

When the model has no reference image (yet) in the sibling skill, the diagram is synthesised from `config.json` + `modeling_*.py` alone. Set `source_basis.image` to `null` and explain in `note`.

## 2. How many diagrams per model

**Baseline: 1 diagram per model**, always. The block-level outer view (Embed → blocks → norm → LMHead). Even a fully-standard transformer earns this baseline diagram so the skill is uniformly addressable by model name.

**Add module-detail diagram(s)** only for modules listed in section 3 below as "auto-earn detail". Modules listed as "never earn detail" do not justify a second file no matter how often they appear.

**Hard cap: 3 diagrams per model.** If more than 3 module-detail bars are tripped, the splits are too fine — fold related modules into one detail diagram or move secondary detail to `## Notes`.

## 3. Module catalogue

The matrix below is the operational rule. When authoring a new model, list the modules it uses (from `config.json` + `modeling_*.py`), look each one up here, and tally diagrams.

### Auto-earn a detail diagram

| Module | Why it earns a detail diagram |
|---|---|
| **MLA (Multi-head Latent Attention)** | 3+ branch fan-out (`c_Q`, `c_KV`, `k_rope_pre`), sub-dim RoPE injection, KV-cache compression. DeepSeek family. |
| **Shared + routed parallel MoE** | Routed top-k experts and 1+ shared experts feed back in parallel — different computation graph from vanilla top-k. DeepSeek MoE (V3/V3.2/V4), Hunyuan-A13B. |
| **Cross-modal fusion (VLM)** | How vision tokens enter the LLM (early fusion / late fusion / cross-attention) is the entire structural identity of a VLM. Qwen3-VL, Kimi-VL. |
| **Hybrid attention** | Multiple parallel attention paths in one block (e.g. linear + softmax, attn + state-space). MiniMax-style hybrids if/when present, Jamba-style hybrids. |
| **MTP / multi-token prediction heads** | Distinct head-side computation graph alongside the main LMHead. DeepSeek V3/V4 trained with MTP. |
| **Non-trivial router** | Auxiliary-loss-free balancing, expert-choice routing, or any router that changes the per-token compute path. DeepSeek-MoE V3-style balancing. (Standard top-k token-choice does NOT trip this.) |

### Never earn a detail diagram

| Module | Why not |
|---|---|
| Vanilla MHA / GQA / MQA | `W_Q`/`W_K`/`W_V` three-way is universally known, no surprises in the forward. |
| Standard top-k MoE (no shared expert, no routing innovation) | `router → top-k → expert → weighted sum` is the textbook MoE; no information beyond the block-level pill. |
| SwiGLU / GeGLU / GLU FFN variants | Activation choice; no structural variation worth its own diagram. |
| RMSNorm / LayerNorm / pre-norm ordering | Trivial. |
| Full-head RoPE | All dimensions rotated; no sub-dim trick to draw. |
| Sliding window / chunked attention | Boundary rule, not topology. Encode in `## Notes`. |

### Conditional

| Module | Condition for earning a detail diagram |
|---|---|
| KV-cache compression | If part of MLA — already covered by the MLA detail. If a non-MLA compression scheme (Quest, H2O, learned eviction) — earns a detail. |
| Vision encoder (VLM) | If stock SigLIP / CLIP / OpenCLIP — do not draw; link out in `## Notes`. If custom (DeepSeek-VL vision encoder, MiniGPT-style perceiver, etc.) — earns a detail. |
| Inference-mode variants of the same module | Earns a detail when the computation graph genuinely differs across modes, not just parameters. Example: MLA-MHA absorption vs MLA-MQA absorption are different graphs sharing one structural identity; either decide on one diagram with both modes annotated, or split into two — never blur. |

## 4. In-scope models

This skill targets text-graph-friendly architectures. Diffusion / video / 3D model architectures are visual-spatial enough that ASCII/Mermaid expresses them poorly — those stay with the image-based sibling skill.

| Scope | Families |
|---|---|
| **In-scope** | DeepSeek V3 / V3.2 / V4, GLM-5, Qwen3 (dense + MoE) / Qwen3.5, Kimi K2 / K2.5, MiniMax M2 / M2.5, Step 3.5 Flash, Hunyuan-A13B, Llama 4 (dense + MoE), Qwen3-VL, Kimi-VL |
| **Out-of-scope** | Z-Image, Wan2.1, Wan2.2, HunyuanVideo, Hunyuan3D-2, FLUX.1 |

When a new in-scope model lands, the author follows the playbook below; no additional approval needed unless the model uses a module not yet in section 3 (in which case extend section 3 first, then author).

## 4a. Closed-vocabulary tags

Four tag spaces are closed and frozen. Authors choose values from these lists; do not invent new ones.

**Lazy-extension rule.** The vocabularies below are an initial seed; expect them to grow over time as new model families land. Extend a vocabulary **only** when a new model genuinely doesn't fit an existing tag — never preemptively add slots for hypothetical future architectures. When extending, **new tag names must follow conventions already used by**, in order of preference:

1. The model's **HF model card / config** description.
2. The model's **official technical report**.
3. **Module class names** in `modeling_*.py`.

Do not coin novel terminology inside this skill.

Three of the four tag spaces are **model-level orthogonal axes** (used in frontmatter + `## Model summary`). Each axis captures one independent decision-affecting dimension; downstream agents can filter on any axis alone (e.g. "all `attention_type=mla` models", "all `ffn_type=moe-shared+routed` models").

### `modality` (model-level)

| Tag | Meaning |
|---|---|
| `text` | Text-only LLM. |
| `text+vision` | VLM (image / video inputs). |
| `text+audio` | Speech / audio-conditioned LLM. |
| `text+vision+audio` | Omni-modal model. |

### `attention_type` (model-level)

| Tag | Meaning |
|---|---|
| `mha` | Standard multi-head attention. |
| `gqa` | Grouped-query attention. |
| `mqa` | Multi-query attention. |
| `mla` | Multi-head latent attention (DeepSeek family). |
| `linear-attn` | Linear / kernel attention (Performer, LinAttn, etc.). |
| `mamba` | State-space model (Mamba family) used in place of attention. |
| `hybrid-{a}+{b}` | Concrete hybrid composition, e.g. `hybrid-mla+linear-attn`. Use only when more than one attention type is interleaved across layers. |
| `chunked-{base}` | Long-context chunking variant of a base attention, e.g. `chunked-gqa`. |

For variants that change only parameters (head count, head dim, RoPE base) of an existing type, keep the base tag and describe the variation in `## Notes`.

### `ffn_type` (model-level)

| Tag | Meaning |
|---|---|
| `dense` | Dense feed-forward only (SwiGLU / GeGLU / GLU / vanilla MLP). |
| `moe-routed` | Top-k routed MoE, no shared expert (Qwen3-MoE, Mixtral-style). |
| `moe-shared+routed` | Top-k routed experts **plus** one or more shared experts always-on in parallel (DeepSeek V3 family, Hunyuan-A13B). |

For models that mix dense FFN at some layers and MoE at others (e.g. DeepSeek V3.2 with dense FFN at layers 1–3), `ffn_type` records the **dominant** variant; the exception is noted in `## Notes` and reflected in the `## Modules` table with a `5*`-style alternating row.

### `type` (per-module, used in `## Modules` table)

| Tag | Meaning |
|---|---|
| `embed` | Token / position embedding layer at the front of the network. |
| `norm` | Any normalization (RMSNorm, LayerNorm). |
| `attn` | Any attention block (MHA, GQA, MQA, MLA, hybrid attention). |
| `ffn-dense` | Dense feed-forward block (SwiGLU, GeGLU, GLU, vanilla MLP). |
| `ffn-moe` | Any MoE feed-forward block (with or without shared experts; routed top-k or expert-choice). |
| `fusion` | Cross-modal fusion in VLMs (cross-attention into LLM, projector / merger / Q-Former / perceiver). |
| `vision-encoder` | Vision encoder tower in a VLM (SigLIP / CLIP / custom). |
| `head` | Output projection layer (`LMHead` for LLMs, classification head, etc.). |
| `mtp` | Multi-token prediction head (DeepSeek-style auxiliary prediction). |

When a module behaves as more than one of these (rare), pick the *dominant* compute type and explain in `## Notes`. Don't list multiple tags.

## 5. Authoring playbook (per new model)

1. **Identify modules used.** Read `config.json` and skim `modeling_*.py` (`<Model>DecoderLayer.forward`, attention class, FFN/MoE class, any head module).
2. **Look up each module in section 3.** Tally: 1 (baseline block diagram) + 1 per "auto-earn" module hit + conditional ones per their condition. Verify total ≤ 3.
3. **Pick the model's three model-level tags** — `modality`, `attention_type`, `ffn_type` — from section 4a's closed vocabularies. Each axis is independent; do not encode them into a single combined tag.
4. **For each diagram to produce**, follow the sourcing rule in section 1.
5. **Write the `.md` file** per `mermaid-style-guide.md`'s file-structure rule. For block-level diagrams that means the full section order including `## Model summary` and `## Modules`; for module-detail diagrams, omit those two sections (the file *is* one module's interior).
6. **In `## Modules`**, every row carries a closed-vocabulary `type` tag from section 4a; the `Count` column is the number of times the module is instantiated in the full forward pass (1 for one-shot modules at model boundaries, `n_layers` for per-block modules). When a position alternates between two implementations (e.g. dense FFN for layers 1–3 + MoE FFN for the rest), list both as separate rows and note the position-to-layer mapping below the table.
7. **Cross-link** the block diagram to detail diagrams via `[[detail-id]]` in `## Notes`, and the detail diagrams back to the block diagram via `[[block-id]]`.
8. **Self-check (once resolver is implemented)**: resolver matches the model by all expected aliases; ids match filenames; no id collisions; every `modality` / `attention_type` / `ffn_type` / `type` value is in the closed vocabulary.

## 6. Extending this policy

When a model uses a module not listed in section 3:

1. Decide which bucket it belongs to (auto-earn / never / conditional) using the existing rationale style (does it have ≥3 fan-out? does its name fail to imply its dataflow? does it have multiple computation modes?).
2. Add it to section 3 in the same commit as the new model's diagram(s).
3. If the user (skill owner) is uncertain, ask before committing — section 3 is the durable policy and should not drift silently.
