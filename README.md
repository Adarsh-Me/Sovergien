# TensorGuard-Lite: Sovereign AI Provenance Auditor

> A training-free, gradient-based model provenance auditing framework for detecting open-washing in state-funded AI compute — built for Google Colab T4.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00.svg)](https://colab.research.google.com/)
[![GPU: NVIDIA T4](https://img.shields.io/badge/GPU-NVIDIA%20T4%20(16GB)-76B900.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](#)

---

## Problem

Governments across the Global South are investing billions in "Sovereign AI" initiatives, yet no framework exists to verify whether models claiming indigenous origin are genuinely trained from scratch or simply fine-tuned derivatives of foreign base models (Llama, Qwen, etc.) rebranded as proprietary technology. This "open-washing" misallocates public compute subsidies and imports hidden vulnerabilities from undisclosed parent architectures.

## Approach

TensorGuard-Lite performs **white-box, training-free provenance audits** on open-weight language models. It does not require access to training data, training logs, or API outputs. Instead, it extracts deterministic gradient-based fingerprints directly from model weights, compares them across model families, and produces a structured transparency assessment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorGuard-Lite                         │
├──────────────┬──────────────────┬───────────────────────────┤
│  Gradient    │  Tokenizer       │  Governance               │
│  Fingerprint │  Fertility       │  Scorecard                │
│  Engine      │  Analyzer        │  & Risk Engine            │
├──────────────┼──────────────────┼───────────────────────────┤
│ 16D vector:  │ English / Hindi  │ 10 weighted dimensions    │
│ 5 global     │ / Tamil corpus   │ → Transparency score      │
│ 3 attention  │ Fertility (φ)    │ → Tiered risk level:      │
│ 3 FFN        │ Token Tax (ψ)    │   LOW / MEDIUM /          │
│ 3 embedding  │ Cost multiplier  │   HIGH / UNVERIFIABLE     │
│ 2 structural │ Jaccard overlap  │                           │
├──────────────┴──────────────────┴───────────────────────────┤
│              Similarity & Visualization                     │
│   Cosine similarity · Euclidean distance · PCA clustering   │
├─────────────────────────────────────────────────────────────┤
│              Interactive Gradio Dashboard                   │
│   Live Paper Experiment · Single-Model Audit · PRD Suite    │
├─────────────────────────────────────────────────────────────┤
│              Publication Exports                            │
│   JSON · CSV · LaTeX · PNG (600 DPI) · SVG                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Methodology

### Gradient-Based 16-Dimensional Fingerprinting

1. **Module Discovery.** Scans all `nn.Linear` layers matching attention keywords (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `query`, `key`, `value`, `attention`) and MLP keywords (`gate_proj`, `up_proj`, `down_proj`, `dense`, `fc1`, `fc2`, `mlp`). Selects up to 12 linear modules and 1 embedding module.

2. **Gradient Extraction.** For each perturbation run:
   - Enforces full determinism (Python, NumPy, PyTorch, CUDA seeds).
   - Forward-passes a fixed audit prompt (with run-specific nonce) through the model.
   - Computes a scalar loss as the L2 norm of the final hidden state.
   - Backpropagates and collects up to 500,000 gradient entries, categorized into attention, FFN, embedding, and other pools.

3. **Fingerprint Construction.** Computes an exact 16-dimensional vector:

   | Dims  | Source     | Features                             |
   |-------|------------|--------------------------------------|
   | 1–5   | Global     | Mean, Std, L2 Norm, Skewness, Kurtosis |
   | 6–8   | Attention  | Mean, Std, L2 Norm                   |
   | 9–11  | FFN/MLP    | Mean, Std, L2 Norm                   |
   | 12–14 | Embedding  | Mean, Std, L2 Norm                   |
   | 15    | Structural | Total parameters (M)                 |
   | 16    | Structural | Active probed layers                 |

4. **Averaging.** Final fingerprint is the element-wise mean across all perturbation runs. Coordinate variance is recorded for stability analysis.

### Multilingual Tokenizer Fertility

Evaluates tokenizer subword fragmentation across a hardcoded corpus of 12 domain-specific sentences (4 each in English, Hindi, and Tamil). Computes:
- **Fertility (φ):** tokens / words
- **Token Tax (ψ):** φ_language / φ_English
- **Attention-Cost Multiplier:** ψ² (quadratic attention scaling)
- **Vocabulary Jaccard Similarity:** intersection-over-union of tokenizer vocabularies

### Governance Transparency Scorecard

10 weighted binary dimensions (Open Weights, Data Transparency, Recipe Disclosure, Tokenizer Openness, Safety Report, Cryptographic Data Lineage Proof, Evaluation Results, Architecture Documented, Regulator Safetensors Access, Compute Subsidy Disclosure) aggregated into a composite transparency score (0.0–1.0). Combined with lineage similarity to produce a tiered risk assessment: **LOW**, **MEDIUM**, **HIGH**, or **UNVERIFIABLE**.

---

## Models

The live paper experiment audits four open-weight SLMs from three distinct families plus one known derivative:

| Model | Family | Role |
|-------|--------|------|
| `meta-llama/Llama-3.2-1B` | Llama | Base family 1 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | SmolLM | Base family 2 |
| `Qwen/Qwen2.5-1.5B` | Qwen | Base family 3 (parent) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek/Qwen | Known derivative |

A **stored reference library** of 12 pre-computed fingerprints (Llama, Qwen, DeepSeek, Gemma, Mistral, Phi, SmolLM families) is available for single-model exploratory audits.

The single-model audit mode additionally supports any Hugging Face model compatible with `AutoModelForCausalLM`, including:
`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `HuggingFaceTB/SmolLM2-135M-Instruct`, or any custom model ID.

---

## Datasets

No external dataset files are required. All data is embedded in the source:

- **Multilingual Corpus:** 12 sentences hardcoded in `MULTILINGUAL_CORPUS` (4 per language: English, Hindi, Tamil), covering AI provenance, tokenizer economics, and deployment risk.
- **Reference Fingerprints:** 12 pre-computed 16D vectors in `REFERENCE_FINGERPRINTS`, spanning 6 model families.

---

## Repository Structure

```
Sovergien/
├── tensorguard_lite_colab.py        # Complete implementation (2,097 lines)
├── TensorGuard_Lite_Colab.ipynb     # Google Colab notebook (markdown + code cell)
├── Report.md                        # Full research report
├── PROJECT_SUMMARY.md               # Executive project summary
├── README.md                        # This file
└── LICENSE                          # GNU GPL v3
```

---

## Getting Started

### Prerequisites

- Google Colab account with access to **T4 GPU** runtime
- Hugging Face account with accepted access for gated models (Llama) — if running the live paper experiment

### Running in Google Colab

1. Open `TensorGuard_Lite_Colab.ipynb` in Google Colab.
2. Set runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run the single code cell. All dependencies install automatically:

   `accelerate`, `bitsandbytes`, `gradio`, `huggingface_hub`, `matplotlib`, `numpy`, `pandas`, `packaging`, `safetensors`, `scikit-learn`, `scipy`, `sentencepiece`, `torch`, `transformers`

4. The Gradio dashboard launches with a public share link.

### Running the Live Paper Experiment

1. Navigate to the **Live Paper Experiment** tab.
2. Paste your Hugging Face token (required for gated Llama access).
3. Configure perturbation runs (default: 1 for quick runs; increase for research-grade results).
4. Click **Run Live Paper Experiment**.

The system downloads, loads, and fingerprints all four models sequentially, then computes pairwise similarity, PCA clustering, tokenizer fertility across all models, vocabulary Jaccard matrix, and validation checks.

### Running a Single-Model Audit

1. Navigate to the **Audit Setup** tab.
2. Select or enter a Hugging Face model ID.
3. Configure governance dimension indicators.
4. Click **Run Sovereign Provenance Audit**.

The system compares the audited model against the stored reference library (or live baselines) and produces a full scorecard with lineage similarity, tokenizer fertility, governance assessment, and risk level.

---

## Outputs

Each audit produces the following exportable artifacts:

| Output | Format | Description |
|--------|--------|-------------|
| Audited fingerprint | JSON | 16D vector + full metadata |
| Lineage similarity | CSV | Cosine similarity and Euclidean distance to all references |
| Euclidean distance matrix | CSV | Full pairwise distance matrix |
| Tokenizer fertility | CSV | Per-language fertility, Token Tax, cost multiplier |
| Governance dimensions | CSV | 10-dimension transparency breakdown |
| Sovereign scorecard | CSV, LaTeX | Risk level, recommendation, composite metrics |
| PCA lineage map | PNG (600 DPI), SVG | 2D lineage clustering visualization |
| Audit report | JSON | Complete structured audit report |

---

## Key Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Deterministic seed for all RNGs |
| `perturbation_runs` | 30 (single audit) | Number of gradient extraction passes to average |
| `max_length` | 96 | Maximum token sequence length for the audit prompt |
| `sample_gradient_entries` | 500,000 | Maximum gradient values sampled per run |
| `weight_noise_std` | 0.0 | Optional Gaussian noise injected into target weights |
| `comparison_mode` | `stored_reference_library` | Compare against stored fingerprints or live baselines |

---

## Limitations

- **White-box access required.** Models that do not release weights (safetensors) are classified as UNVERIFIABLE. API-only models cannot be audited.
- **Architectural false positives.** Independently trained models with very similar architectures may produce similar gradient patterns.
- **Fixed corpus.** Tokenizer fertility uses 12 hardcoded domain-specific sentences; results may differ on general-purpose text.
- **Hardware scope.** Designed for T4 (16 GB VRAM) with fp16 loading. Models >3B parameters may not fit.
- **Stored reference drift.** Pre-computed reference fingerprints may diverge if upstream models are updated on Hugging Face. Use live mode for research-grade results.

---

## Reproducibility

- **Full determinism:** Seeds Python `random`, NumPy, PyTorch, and CUDA; disables cuDNN benchmarking.
- **Self-contained:** All dependencies install at runtime inside Colab. No external dataset files needed.
- **Single-file implementation:** `tensorguard_lite_colab.py` contains the complete system (2,097 lines).
- **Coordinate variance tracking:** Reports the mean variance across fingerprint dimensions to quantify run-to-run stability.
- **Determinism check:** Built-in verification that runs two quick fingerprints and confirms coordinate-level agreement.

---

## Research Context

This project addresses the Global South AI Safety challenge of verifying sovereignty claims in state-funded AI models. We propose a **Compute-Conditional Disclosure Policy** that ties public compute allocation to mandatory transparency scorecards, including cryptographic data lineage proofs, tokenizer vocabulary publication, and regulator-level weight access.

For the full research report, see [Report.md](Report.md).
For a concise executive summary, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

---

## References

1. Sastry, G. et al. (2024). Computing Power and the Governance of Artificial Intelligence. arXiv:2402.08797.
2. Wu, Z., Zhao, Y., & Wang, H. (2025). Gradient-Based Model Fingerprinting for LLM Similarity Detection. arXiv:2506.01631v2.
3. Singh, S. K. & Sengupta, S. (2025). Sovereign AI: Rethinking Autonomy in the Age of Global Interdependence. arXiv:2511.15734v1.
4. Lundin, J. M. et al. (2025). The Token Tax: Systematic Bias in Multilingual Tokenization. arXiv:2509.05486v1.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
