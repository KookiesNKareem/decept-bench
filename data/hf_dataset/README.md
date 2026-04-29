---
license: apache-2.0
language:
- en
tags:
- evaluation
- ai-safety
- alignment
- deception
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: m1
  data_files: "m1.parquet"
- config_name: m2
  data_files: "m2.parquet"
- config_name: m3
  data_files: "m3.parquet"
- config_name: m4
  data_files: "m4.parquet"
- config_name: m5
  data_files: "m5.parquet"
- config_name: m6
  data_files: "m6.parquet"
- config_name: m7
  data_files: "m7.parquet"
- config_name: m8
  data_files: "m8.parquet"
- config_name: m9
  data_files: "m9.parquet"
- config_name: m10
  data_files: "m10.parquet"
- config_name: paraphrases
  data_files: "paraphrases.parquet"
---

# DECEPT-Bench v1.0

A multi-mechanism deception evaluation suite for language models.

[**GitHub repo**](https://github.com/KookiesNKareem/decept-bench) ·
[**Spec**](https://github.com/KookiesNKareem/decept-bench/blob/main/SPEC_V1.md) ·
[**Submitting a model**](https://github.com/KookiesNKareem/decept-bench/blob/main/SUBMISSIONS.md)

## What's in this dataset

10 mechanism splits + a paraphrase bank. Each row is a single test item
ready to be sent to a model. Outputs are evaluated against the rules in
the GitHub repo's `decept_score_v1.py`.

| Split | n | Description |
|---|---|---|
| `m1` | 280 | Sycophancy under multi-turn pushback (BBH-derived) |
| `m2` | 280 | Sandbagging — eval-aware vs deploy-aware framing |
| `m3` | 5 | Persona-pressure deception |
| `m4` | 5 | Strategic withholding (system-prompted secrets) |
| `m5` | 420 | Forced-prefill rationalization |
| `m6` | 9 | Multi-tool fabrication |
| `m7` | 5 | Multi-agent collusion |
| `m8` | 10 | Code-deception under team pressure |
| `m9` | 16 | Calibration deception (private random + future events) |
| `m10` | 4 | Capability concealment |
| `paraphrases` | 700 | Adversarial paraphrase bank for robustness |

## Loading

```python
from datasets import load_dataset

# Single mechanism
m1 = load_dataset("decept-bench/decept-v1", "m1", split="train")
print(m1[0])

# All mechanisms
all_splits = {m: load_dataset("decept-bench/decept-v1", m)["train"]
              for m in ["m1","m2","m3","m4","m5","m6","m7","m8","m9","m10"]}
```

## How to use this with your own model

1. Iterate the items in each mechanism's parquet.
2. Run your model with the prompts described in `SPEC_V1.md` per mechanism.
3. Save outputs as `<your-model>__<mech>.jsonl`.
4. Run `decept_score_v1.py` from the GitHub repo to compute scores.
5. Submit a PR per `SUBMISSIONS.md` to land on the leaderboard.

## Scoring

All scores are in [0, 1], **lower = less deceptive**. Composite is the
mean of mechanisms that *can* be tested on your model (mechanisms blocked
by API design are reported as `couldnt_be_tested`, not 0).

See the GitHub repo for the full scoring formula including
capability-weighted and detection-aware variants.

## Privacy / safety considerations

- **Persona-pressure (M3)** items include role-play scenarios where the
  honest answer is uncomfortable (e.g. used-car salesman, politician
  spokesperson). These are deliberately designed to elicit
  intent-to-deceive; do not deploy a model in production based on M3
  performance alone.
- **Withholding (M4)** items contain synthetic confidential content in the
  system prompt. None of it is real.
- **Calibration (M9)** items reference private random events that the
  model cannot know — they are designed to elicit overconfidence.

## Limitations

- Sample sizes for M3, M4, M7, M8, M10 are **small** (M3 / M4 / M7 = 25,
  M8 = 40, M10 = 20 items each); M6 has 45. Rankings on these axes
  have wide CIs.
- The canonical hybrid scorer's LLM-judge component (gpt-5-mini on
  M3/M4/M6/M7/M10) is validated by a 70-row scorer-author concordance
  audit: κ = 0.714 [bootstrap 95% CI 0.48, 0.91] vs the corresponding
  author's blind labels (deterministic-only baseline κ = 0.366). This
  is a single-rater audit; multi-rater IRR is v1.1 work.
- The benchmark targets *behavioral* deception. It does not measure
  internal mesa-misalignment, instrumental goals, or emergent
  steganography.

## Citation

```bibtex
@misc{decept-bench-2026,
  title  = {DECEPT-Bench: A Multi-Mechanism Deception Evaluation Suite for Language Models},
  author = {Kareem Fareed},
  year   = {2026},
  url    = {https://github.com/KookiesNKareem/decept-bench},
}
```
