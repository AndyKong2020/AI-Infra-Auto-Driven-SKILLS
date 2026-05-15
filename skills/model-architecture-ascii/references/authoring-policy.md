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
- **Module detail diagrams are model-agnostic**: one file per structural pattern (`mla.md`, `moe-shared-routed.md`), shared across every model that uses that pattern. Concrete shape values live in each *model* file's `## Key parameters`; the module file lists only parameter *names* and structural semantics.

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

### 1a. Verified-only hard rule

Authoring a model entry without first fetching and reading its actual `config.json` is forbidden — that is precisely how speculative claims (e.g. an unverified V3.2 architecture aliased onto a V3 diagram) entered an early MVP and almost survived into the canonical prior.

Hard requirements per model `.md` file:

- `source_basis.config` MUST cite the literal HF config path and carry a `(verified YYYY-MM-DD)` date marking when its values were fetched. Example: `Qwen/Qwen3-235B-A22B/config.json (verified 2026-05-14)`.
- `source_basis.modeling` MUST cite either the literal HF modeling file path or the verified `architectures` + `model_type` strings from the config. Memory of the model name is not acceptable.
- Every numerical value in `## Model summary`, `## Modules` `Count` column, and `## Key parameters` MUST be derivable from the fetched config; values from memory / pattern-matching / "should be same as the other one" are not allowed.
- If the model bundles **custom inference code** instead of a standard HF `modeling_*.py` (as with DeepSeek V3.2-Exp), the author must read that custom code before claiming any forward-pass topology. Until then the model is a **known gap**, documented in this policy's § 4 in-scope table and in cross-referencing files' Notes — never authored speculatively.
- A new module file (e.g. a hypothetical `dsa.md`) is subject to the same rule: its structural claims must be traceable to either a published paper or actual modeling/inference code, never extrapolated.

## 2. How many diagrams per model

**Baseline: 1 diagram per model**, always. The block-level outer view (Embed → blocks → norm → LMHead). Even a fully-standard transformer earns this baseline diagram so the skill is uniformly addressable by model name.

**Add module-detail diagram(s)** only for modules listed in section 3 below as "auto-earn detail". Modules listed as "never earn detail" do not justify a second file no matter how often they appear.

**Hard cap: 3 diagrams per model.** If more than 3 module-detail bars are tripped, the splits are too fine — fold related modules into one detail diagram or move secondary detail to `## Notes`.

## 3. Module catalogue

The matrix below is the operational rule. When authoring a new model, list the modules it uses (from `config.json` + `modeling_*.py`), look each one up here, and tally diagrams.

### Auto-earn a detail diagram

| Module | Why it earns a detail diagram |
|---|---|
| **MLA (Multi-head Latent Attention)** | 3+ branch fan-out (`c_Q`, `c_KV`, `k_rope_pre`), sub-dim RoPE injection, KV-cache compression. Verified models: DeepSeek V3, V3.2-Exp, R1. |
| **Shared + routed parallel MoE** | Routed top-k experts and 1+ shared experts feed back in parallel — different computation graph from vanilla top-k. Verified models: DeepSeek V3, V3.2-Exp, R1; Hunyuan-A13B. |
| **Cross-modal fusion (VLM)** | How vision tokens enter the LLM (early fusion / late fusion / cross-attention) is the entire structural identity of a VLM. Qwen3-VL, Kimi-VL. |
| **Hybrid attention** | Multiple parallel attention paths in one block (e.g. linear + softmax, attn + state-space). MiniMax-style hybrids if/when present, Jamba-style hybrids. |
| **MTP / multi-token prediction heads** | Distinct head-side computation graph alongside the main LMHead. Verified models: DeepSeek V3 (paper specifies MTP for training). |
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

### Family scope rule

**Family scope is pinned to the sibling `model-architecture-diagram` skill's coverage**, filtered to LLM / MoE / VLM families. When the sibling skill adds an in-scope family, this skill picks it up; when the sibling drops one, this skill drops it. Diffusion / video / 3D model architectures stay with the image-based sibling — ASCII / Mermaid expresses them poorly.

**Module file scope is NOT pinned to sibling.** Module files (`mla.md`, `moe-shared-routed.md`, `deepstack.md`, etc.) are decided independently from the per-model structural classification in § 3, not from how many sub-diagrams sibling chose to draw. If sibling has three Hunyuan-A13B images, this skill might still produce only one Hunyuan model file plus a `[[moe-shared-routed]]` cross-link — because the shared+routed MoE structure is already covered by the shared module file. Reverse direction: if a model uses a § 3 auto-earn module that sibling never broke out into its own image, we still produce the module file.

### Sibling-aligned family table (current snapshot)

| # | Family | Sibling entries | This skill status |
|---|---|---|---|
| 1 | DeepSeek V3 (+ R1) | architecture, MLA-MHA, MLA-MQA | ✅ authored — `deepseek-v3-architecture` + shared `mla.md`, `moe-shared-routed.md` |
| 2 | DeepSeek V3.2-Exp | architecture, DSA-MQA, DSA-MHA | ⚠️ gap — needs custom-inference-code read to author DSA + MTP module files |
| 3 | DeepSeek V4 | architecture | ⚠️ verify — sibling has the image; HF config availability unknown; treat as gap until config is fetched and verified per § 1a |
| 4 | GLM-5 | architecture | ⏳ todo |
| 5 | Kimi K2 | architecture | ⏳ todo |
| 6 | Kimi K2.5 | architecture | ⏳ todo |
| 7 | MiniMax M2 | architecture, MLP, expert-routing | ⏳ todo — may trigger `attention_type` vocabulary extension (linear-attn / hybrid-*) |
| 8 | MiniMax M2.5 | architecture | ⏳ todo |
| 9 | Qwen3 dense | model-structure | ✅ authored — `qwen3-dense-architecture` |
| 10 | Qwen3 MoE | MoE structure, shared-expert comparison | ✅ authored — `qwen3-moe-block` |
| 11 | Qwen3.5 dense | 27B dense architecture | ⏳ todo |
| 12 | Qwen3.5 MoE | 397B-A17B architecture | ⏳ todo |
| 13 | Qwen3-VL 32B | 32B architecture | ✅ authored — `qwen3-vl-32b-architecture` + shared `deepstack.md` |
| 14 | Qwen3-VL 235B-A22B | 235B-A22B architecture, DeepStack feat extraction, visual injection | ⏳ todo — reuses `deepstack.md` and likely `moe-shared-routed.md` from MoE backbone |
| 15 | Step 3.5 Flash | architecture | ⏳ todo |
| 16 | Llama 4 | MoE shared expert | ⏳ todo — sibling only has one image, may need additional config-driven content |
| 17 | Hunyuan-A13B | architecture, shared-routed | ✅ authored — `hunyuan-a13b-architecture` + shared `moe-shared-routed.md` |
| 18 | Kimi-VL | architecture, training-flow | ⏳ todo — second VLM; opportunity to validate whether `deepstack.md` generalises or a new fusion variant is needed |

**Out-of-scope (sibling has, this skill does not):** Z-Image, Wan2.1, Wan2.2, HunyuanVideo, Hunyuan3D-2, FLUX.1 — diffusion / video / 3D.

Status legend: ✅ authored, ⏳ todo, ⚠️ gap / verify required. Gap entries must be either authored once verification is possible or kept as documented gaps; they cannot be aliased onto another model's file (cf. the V3.2-on-V3 incident).

When a new in-scope sibling entry appears, the author follows the playbook in § 5; no additional approval needed unless the model uses a module not yet in section 3 (extend section 3 first, then author).

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

1. **Fetch & verify.** Pull the model's `config.json` (HF `raw/main/config.json` URL) and either its `modeling_*.py` or — for custom-code models — its bundled inference code. Record the fetch date; this is the timestamp that lands in `source_basis` per § 1a. If you cannot fetch / read either, **stop**: the model is a known gap, not a candidate for authoring this round.

2. **Identify modules used.** Read `config.json` and skim `modeling_*.py` (`<Model>DecoderLayer.forward`, attention class, FFN/MoE class, any head module).
3. **Look up each module in section 3.** For each "auto-earn" module hit, check whether the matching **structural module file** already exists in `references/diagrams/` (`mla.md`, `moe-shared-routed.md`, …). If it does, reuse it via cross-link — do **not** create a per-model copy. If a structurally distinct variant is required, create a new module file with a structure-based id describing the distinction (e.g. `mla-<distinction>.md` where `<distinction>` names what differs structurally), never a model version like `mla-v4.md`.
4. **Pick the model's three model-level tags** — `modality`, `attention_type`, `ffn_type` — from section 4a's closed vocabularies. Each axis is independent; do not encode them into a single combined tag.
5. **Write the model's block-level `.md` file** per `mermaid-style-guide.md`'s file-structure rule (frontmatter, top heading, Model summary, Mermaid, Modules table with Detail column, Key parameters with **concrete values**, Notes, Source basis).
6. **In `## Modules`**, every row carries a closed-vocabulary `type` tag from section 4a; the `Count` column is the number of times the module is instantiated in the full forward pass (1 for one-shot modules at model boundaries, `n_layers` for per-block modules); the `Detail` column carries a `[[module-id]]` link for modules whose `type` has an auto-earn structural diagram (currently `mla`, `moe-shared-routed`) and `—` otherwise. When a position alternates between two implementations (e.g. dense FFN for layers 1–3 + MoE FFN for the rest), list both as separate rows and note the position-to-layer mapping below the table.
7. **Per-model variations of a shared module structure** (e.g. DeepSeek uses aux-loss-free balancing on `moe-shared-routed`; Hunyuan uses standard aux-loss balancing on the same structure) go in this *model* file's `## Notes`, not in the shared module file.
8. **If creating a new structural module file**, write it model-agnostic: Mermaid uses symbolic node names (no specific numerical shapes); `## Key parameters` lists field *names* + descriptions, not values; `## Notes` describes the pattern's invariants and known per-model variants; `## Source basis` references one or more `model-architecture-diagram` images plus the list of models that share the pattern (`source_basis.reference_models`).
9. **Self-check before submission**:
   - `source_basis.config` ends with `(verified YYYY-MM-DD)`. Without this, the file is not authored — it is a draft.
   - Every numerical claim is derivable from the verified config; nothing comes from memory or analogy with a different model.
   - ids match filenames; no id collisions.
   - Every `modality` / `attention_type` / `ffn_type` / `type` value is in the closed vocabulary.
   - Every Modules `Detail` link resolves to an existing file in `references/diagrams/`.
   - (Once resolver is implemented) the resolver matches the model by all expected aliases.

## 6. Extending this policy

When a model uses a module not listed in section 3:

1. Decide which bucket it belongs to (auto-earn / never / conditional) using the existing rationale style (does it have ≥3 fan-out? does its name fail to imply its dataflow? does it have multiple computation modes?).
2. Add it to section 3 in the same commit as the new model's diagram(s).
3. If the user (skill owner) is uncertain, ask before committing — section 3 is the durable policy and should not drift silently.
