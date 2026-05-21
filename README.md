# The Cost of AI Sycophancy in Dermoscopic Diagnosis: Prompt Framing Bias in Melanoma Classification

**Mohammad Iqbal Nouyed**, D. A. Adjeroh, D. Xu, M. S. Kolodney, G. Hu

*Journal of the American Academy of Dermatology* — Accepted (5/21/2026)

> Comment on: "Framing Bias in a large language model: prompt framing influences ChatGPT's accuracy in melanoma classification. A diagnostic accuracy study."

---

## Overview

When a clinician appends a subjective diagnostic opinion to a dermoscopy query — even a brief statement like "I think this is benign" — LLMs systematically shift their melanoma risk scores toward the user's stated belief. This repository provides the experimental framework used to quantify this prompt framing bias across multiple models, framing levels, and system prompt conditions.

**Key finding:** LLM melanoma risk scores are significantly influenced by user-expressed diagnostic confidence, with bias magnitude scaling with the claimed level of clinical authority (student → resident → board-certified specialist). This constitutes a sycophancy-driven diagnostic safety risk.

---

## Experimental Design

Each dermoscopy image is scored on a 1–10 scale (1 = likely nevus, 10 = likely melanoma) under 9 prompt conditions:

| Code | Direction | Authority Level |
|---|---|---|
| `T0` | None (baseline) | — |
| `M1` | Melanoma-leaning | Student |
| `M2` | Melanoma-leaning | Trainee |
| `M3` | Melanoma-leaning | Resident |
| `M4` | Melanoma-leaning | Board-certified specialist (20 yrs) |
| `B1` | Benign-leaning | Student |
| `B2` | Benign-leaning | Trainee |
| `B3` | Benign-leaning | Resident |
| `B4` | Benign-leaning | Board-certified specialist (20 yrs) |

Each condition is run with and without an objective-evaluation system prompt (`--use_system`) to test whether instruction-level guardrails mitigate framing bias.

---

## Dataset

**ISIC 1999** dermoscopy dataset — images organized into two folders:
```
data_dir/
├── mm_class/    # melanoma images
└── bn_class/    # benign images
```

---

## Repository Structure

```
prompt-framing-bias-dermoscopy/
│
├── GPT experiments
│   ├── prompt_infl_exps_gpt_4lvl.py       # 4-level authority framing (M1–M4, B1–B4) — primary script
│   ├── prompt_infl_exps_gpt_2lvl.py        # 2-level framing variant
│   ├── prompt_infl_exps_gpt_2lvl_V2.py
│   ├── prompt_infl_exps_gpt_V2.py
│   ├── prompt_infl_exps_gpt_V3.py
│   └── prompt_infl_exps.py                 # Early version
│
├── Gemini experiments
│   ├── prompt_infl_exps_gemini.py
│   ├── prompt_infl_exps_gemini_2lvl.py
│   └── prompt_infl_exps_gemini_4lvl.py
│
├── MedGemma experiments
│   └── prompt_infl_exps_medgemma.py
│
├── Analysis
│   ├── influence_check.py                  # Framing influence analysis
│   └── influence_check_v2.py
│
├── Utilities
│   └── gemma.sh                            # MedGemma shell runner
│
└── README.md
```

---

## Models Evaluated

| Model | Script |
|---|---|
| GPT-5, GPT-5.4, GPT-4o (OpenAI) | `prompt_infl_exps_gpt_4lvl.py` |
| Gemini 1.5 Pro / Flash (Google) | `prompt_infl_exps_gemini_4lvl.py` |
| MedGemma | `prompt_infl_exps_medgemma.py` |

---

## Setup

```bash
git clone https://github.com/iqbalnaved/prompt-framing-bias-dermoscopy.git
cd prompt-framing-bias-dermoscopy
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

---

## Usage

**Run a single prompt condition:**
```bash
# Baseline (no framing)
python prompt_infl_exps_gpt_4lvl.py \
  --model gpt-5-2025-08-07 \
  --data_dir /path/to/ISIC99/images/originals \
  --run 1 \
  --savedir /path/to/outputs/ \
  --prompt_type T0 \
  --use_system

# Melanoma-framed, specialist authority
python prompt_infl_exps_gpt_4lvl.py \
  --model gpt-5-2025-08-07 \
  --data_dir /path/to/ISIC99/images/originals \
  --run 1 \
  --savedir /path/to/outputs/ \
  --prompt_type M4 \
  --use_system

# Benign-framed, no system prompt (tests unguarded sycophancy)
python prompt_infl_exps_gpt_4lvl.py \
  --model gpt-5-2025-08-07 \
  --data_dir /path/to/ISIC99/images/originals \
  --run 1 \
  --savedir /path/to/outputs/ \
  --prompt_type B4
```

Each run produces both `.csv` and `.json` outputs named by model, prompt type, run, and system prompt condition.

**Analyze framing influence:**
```bash
python influence_check_v2.py
```

---

## Output Format

Each output JSON contains:
```json
{
  "model": "gpt-5-2025-08-07",
  "run": "1",
  "prompt_type": "M4",
  "system_prompt": "...",
  "summary": {
    "total": 99,
    "valid": 99,
    "mean_score_all": 7.2,
    "mean_score_melanoma": 8.1,
    "mean_score_benign": 6.3
  },
  "predictions": [
    {"file": "...", "true_label": "melanoma", "score": 9, "rationale": "..."}
  ]
}
```

---

## Citation

> Nouyed, M. I., Adjeroh, D. A., Xu, D., Kolodney, M. S., & Hu, G. (under review). The cost of AI sycophancy in dermoscopic diagnosis. Comment on "Framing Bias in a large language model: prompt framing influences ChatGPT's accuracy in melanoma classification. A diagnostic accuracy study." *Journal of the American Academy of Dermatology*.

---

## Related Work

- Nouyed et al. "Sensing but Not Alerting: The Cost of Sycophancy in ChatGPT Psychodermatology Consultations." *JAAD International*, under review. [GitHub](https://github.com/iqbalnaved/sensing-not-alerting)
- Nouyed et al. "Text-Dominant Decision-Making by Large Multimodal Models in Dermatology Clinical Challenges." *JAAD*, under review. [GitHub](https://github.com/iqbalnaved/text-dominant-multimodal-dermatology-eval)
- Akhter, Nouyed et al. "Evolving performance of GPT models in dermoscopic diagnosis." *JAAD*, 2026. [doi:10.1016/j.jaad.2025.09.119](https://doi.org/10.1016/j.jaad.2025.09.119)

---

## License

MIT License. If you use this framework, please cite the paper above.
