# TensorGuard-Lite: Project Summary

## Problem

State governments across the Global South are investing billions in "Sovereign AI" initiatives, yet no mechanism exists to verify whether domestically branded models are genuinely indigenous or "open-washed" fine-tunes of foreign base architectures. This accountability gap enables misallocation of public compute subsidies and imports hidden vulnerabilities from undisclosed parent models.

## Approach

**TensorGuard-Lite** is a training-free, white-box provenance auditing framework that extracts deterministic gradient-based fingerprints from open-weight language models and compares them to detect architectural lineage. It runs entirely on a Google Colab T4 GPU (16 GB VRAM) and requires no access to training data, training logs, or API outputs — only the model weights in safetensors format.

## Methodology

The system implements three complementary audit probes:

1. **Gradient Fingerprinting.** Discovers attention, MLP, and embedding modules in a model's architecture, performs multiple deterministic forward-backward passes using a fixed audit prompt, and collects gradient statistics. These are aggregated into an exact **16-dimensional fingerprint vector** (5 global stats, 3 attention, 3 FFN, 3 embedding, 2 structural features). The final fingerprint is the element-wise mean across perturbation runs, ensuring stability.

2. **Multilingual Tokenizer Fertility.** Measures tokenizer subword fragmentation across English, Hindi, and Tamil using a hardcoded domain-specific corpus (12 sentences). Computes the **Token Tax** ratio (non-English fertility relative to English) and an approximate **attention-cost multiplier** (Token Tax²). Also computes tokenizer **vocabulary Jaccard similarity** between models to detect vocabulary-level inheritance.

3. **Governance Transparency Scorecard.** Evaluates 10 weighted binary dimensions (Open Weights, Data Transparency, Recipe Disclosure, Tokenizer Openness, Safety Report, Cryptographic Data Lineage Proof, Evaluation Results, Architecture Documented, Regulator Safetensors Access, Compute Subsidy Disclosure) to produce a composite transparency score (0.0–1.0) and a **tiered risk assessment** (Low / Medium / High / Unverifiable).

## Models Audited

| Model | Family | Role |
|-------|--------|------|
| `meta-llama/Llama-3.2-1B` | Llama | Base family 1 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | SmolLM | Base family 2 |
| `Qwen/Qwen2.5-1.5B` | Qwen | Parent model |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek/Qwen | Known derivative |

A stored reference library of 12 pre-computed fingerprints across 6 families (Llama, Qwen, DeepSeek, Gemma, Mistral, Phi, SmolLM) is available for single-model audits.

## Key Features

- **Training-free audit** — no fine-tuning, no training data access needed
- **Deterministic fingerprints** — fully seeded execution with coordinate variance tracking
- **16-dimensional gradient fingerprint schema** with exact, documented structure
- **Pairwise cosine similarity and Euclidean distance** between fingerprints
- **PCA lineage clustering** visualization (600 DPI PNG + SVG)
- **Multilingual token tax** analysis across English, Hindi, and Tamil
- **Tokenizer vocabulary Jaccard similarity** for vocabulary inheritance detection
- **10-dimension weighted governance scorecard** with tiered risk assessment
- **Interactive Gradio dashboard** with live paper experiment and single-model audit modes
- **Publication-ready exports** in JSON, CSV, LaTeX, PNG, and SVG

## Outputs

Each audit produces: a 16D fingerprint with full metadata (JSON), lineage similarity tables (CSV), a full Euclidean distance matrix (CSV), tokenizer fertility breakdown (CSV), governance scorecard (CSV + LaTeX), a PCA lineage map (PNG + SVG), and a structured audit report (JSON).

## Key Findings

- **Lineage detection works.** The gradient fingerprinting method clusters the known derivative (DeepSeek-R1-Distill-Qwen-1.5B) with its parent (Qwen2.5-1.5B) while separating cross-family models (Llama, SmolLM).
- **White-box access is the bottleneck.** Models that do not release weights are classified as UNVERIFIABLE, rendering technical auditing moot regardless of fingerprinting accuracy.
- **Token Tax is real.** Non-English languages processed through English-centric tokenizers exhibit materially inflated fertility rates, translating to quadratically higher attention compute costs.

## Limitations

- Requires white-box weight access (safetensors); API-only models cannot be audited.
- Independently trained models with similar architectures may produce similar gradient patterns (false positive risk).
- Tokenizer fertility corpus is small (12 hardcoded sentences) and domain-specific.
- Designed for T4 GPU (16 GB VRAM); models exceeding ~3B parameters may not fit.
- Stored reference fingerprints may drift if upstream models are updated on Hugging Face.

## Policy Proposal

We propose a **Compute-Conditional Disclosure Policy**: any entity receiving state compute subsidies must fulfill a mandatory transparency scorecard including cryptographic data lineage proofs, tokenizer vocabulary publication, regulator-level safetensors access, architecture documentation, and safety evaluation disclosure. The 10-dimension scorecard implemented in this system operationalizes this proposal into an automatable framework.

## Implementation

- **Single Python file:** `tensorguard_lite_colab.py` (2,097 lines)
- **Colab notebook:** `TensorGuard_Lite_Colab.ipynb`
- **Platform:** Google Colab with T4 GPU runtime
- **License:** GNU GPL v3

## Links

- **Full Research Report:** [Report.md](Report.md)
- **Repository:** [GitHub](https://github.com/Adarsh-Me/Sovergien)
