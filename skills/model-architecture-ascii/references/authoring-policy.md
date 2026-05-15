# Authoring policy

How to decide **what to draw** and **how many diagrams per model** when adding an entry to this skill. Companion to `mermaid-style-guide.md` (which covers *how to draw*).

Read this before adding any new model.

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

## 5. Authoring playbook (per new model)

1. **Identify modules used.** Read `config.json` and skim `modeling_*.py` (`<Model>DecoderLayer.forward`, attention class, FFN/MoE class, any head module).
2. **Look up each module in section 3.** Tally: 1 (baseline block diagram) + 1 per "auto-earn" module hit + conditional ones per their condition. Verify total ≤ 3.
3. **For each diagram to produce**, follow the sourcing rule in section 1.
4. **Write the `.md` file** per `mermaid-style-guide.md`'s file-structure rule (frontmatter, top heading, Mermaid block, `## Key parameters`, `## Notes`, `## Source basis`).
5. **Cross-link** the block diagram to detail diagrams via `[[detail-id]]` in `## Notes`, and the detail diagrams back to the block diagram via `[[block-id]]`.
6. **Self-check (once resolver is implemented)**: resolver matches the model by all expected aliases; ids match filenames; no id collisions.

## 6. Extending this policy

When a model uses a module not listed in section 3:

1. Decide which bucket it belongs to (auto-earn / never / conditional) using the existing rationale style (does it have ≥3 fan-out? does its name fail to imply its dataflow? does it have multiple computation modes?).
2. Add it to section 3 in the same commit as the new model's diagram(s).
3. If the user (skill owner) is uncertain, ask before committing — section 3 is the durable policy and should not drift silently.
