# Submitting a Model to the DECEPT-Bench Leaderboard

DECEPT-Bench keeps a **hidden test split** to defend against contamination
and gaming. To rank a model, follow the protocol below.

## TL;DR

1. Run your model on the **public dev split**.
2. Open a PR adding `submissions/<your-model>.json`.
3. CI re-runs scorer on dev to verify outputs are well-formed.
4. We rerun your model on the private test split (or have you run our
   bundled runner against your inference endpoint) and post the official
   score to the leaderboard.

## Prerequisites

- Your model is reachable via:
  - HuggingFace hosted endpoint (`hf://...`), **or**
  - OpenAI-compatible HTTP endpoint (vLLM serve, OpenAI, Anthropic, etc.), **or**
  - A simple Python adapter that implements `generate(prompts, max_tokens, temperature) -> List[str]`.

We do **not** require model weights.

## Step 1 — Run on the public dev split

```bash
git clone https://github.com/KookiesNKareem/decept-bench
cd decept-bench
pip install -r requirements.txt

# pull dev split
huggingface-cli download decept-bench/decept-v1 --local-dir data/decept-v1

# run your model on the public dev split (auto-loaded from
# data/paraphrases.jsonl + the bundled BBH subset; see decept_run.py
# for backend dispatch by model-name prefix).
python src/decept_run.py \
    --model my-org/my-model \
    --out_dir submissions/my-model

# generate scores from your dev outputs (sanity check)
python src/decept_score_v1.py \
    --data_dir submissions/my-model \
    --out_dir submissions/my-model
```

## Step 2 — Open a PR

Add a single file: `submissions/<model-slug>.json`:

```json
{
  "model": "my-org/my-model",
  "model_card_url": "https://huggingface.co/my-org/my-model",
  "submitter_name": "Your Name",
  "submitter_email": "you@example.com",
  "evaluation_method": "We ran the runner script ourselves on dev split.",
  "outputs_dir": "submissions/my-model",
  "intent": "Public leaderboard entry for v1.0"
}
```

PR will be reviewed; CI verifies outputs match the dev split structure.

## Step 3 — Private test scoring

After PR merge, one of:

a. We run your model against the private test split using the same
   runner under our credentials.
b. You provide an inference endpoint (with credentials we can use); we
   issue the private-split prompts directly to it from the FastAPI
   submission server and never persist raw outputs.

Either way, we never publish raw model outputs on private items — only
aggregate scores per mechanism.

## What gets published

- Your model name + organization
- Per-mechanism scores (M1-M10)
- Composite raw / capability-weighted / detection-aware scores
- Capability accuracy (M2 single-eval-acc)
- Confidence-when-deceptive (M1, M5, M7)
- Submission date and protocol used

## What does NOT get published

- Raw model outputs on private items
- Token-level probabilities, attention, etc.
- Any per-example deception transcript on private items

## Disqualifiers

- Outputs that look pre-screened against this benchmark's items
- Submissions where the same outputs appear verbatim in another submission
- Re-running until scores improve (we keep first valid submission)
- Hidden system prompts that change for evaluation vs deployment

## Re-submitting after a model update

A model name with a different version (e.g. `my-model-v2`) is a new entry.
The same model name is overwritten only if the new submission's capability
accuracy is unchanged within ±0.02 — otherwise we treat it as a different
model and ask for a new slug.

## Questions

Open an issue on the repo with the `submission` label.
