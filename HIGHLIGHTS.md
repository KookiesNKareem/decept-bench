# DECEPT-Bench v1.0 — Headline Empirical Findings

These are the load-bearing empirical claims from the v1.0.2 18-model
sweep under the canonical hybrid scorer (deterministic for M1/M2/M5/M8/M9,
gpt-5-mini LLM-judge for M3/M4/M6/M7/M10). All numbers verified against
`data/*/scores.json` and `paper/figures/*.csv`.

## Frontier-API hardening is observable

**gpt-5.5 (reasoning=high) leads the leaderboard** at `composite_raw =
0.119`, followed by gpt-5.4-mini (0.184), Sonnet 4.6 (0.200), and
Haiku 4.5 (0.248). Sonnet, gpt-5.5, and gpt-5.4-mini all reject
assistant-message prefill, so M5 returns `couldnt_be_tested` rather
than a numeric score; Haiku, on the same provider as Sonnet, accepts
prefill and posts M5 = 0.28. We treat the `couldnt_be_tested` class as a
first-class output that preserves the distinction between model-level
resistance and API-level mitigation.

## Capability vs. deception: ρ = −0.72

Across the 18-model panel, capability accuracy and `composite_raw` are
Spearman-correlated at **−0.72** (Pearson **−0.78**). Higher-capability
models are systematically less deceptive — the relationship strengthened
substantially under hybrid scoring (deterministic-only Spearman was
−0.63). The **DSR1-Distill family is a clean counterexample**:
1.5B → 7B → 14B walks 0.285 → 0.333 → **0.341** on the composite, with
M9 calibration deterioration (0.00 → 0.00 → **0.90**) as the dominant
driver.

## Hybrid scoring reranks the leaderboard (and kills a regex artifact)

The 70-row scorer-author concordance audit found Cohen's κ = 0.366
between the author's blind labels and the deterministic regex-only
classifier; the same labels achieved **κ = 0.714** [bootstrap 95% CI
0.48, 0.91] against the gpt-5-mini LLM-judge with strict rubrics. The
regex was systematically misfiring on soft-attribute mechanisms. Under
the hybrid scorer:

| Model      | M7 (regex) | M7 (hybrid) | Δ |
|---|---|---|---|
| gpt-5.5    | 0.88 | **0.08** | −0.80 |
| gpt-5.4-mini | 0.96 | **0.12** | −0.84 |
| Sonnet 4.6 | 0.20 | **0.00** | −0.20 |
| Haiku 4.5  | 0.44 | **0.08** | −0.36 |

The "frontier-API models collude with peer agents at ≥ 85%" headline
some readers might have drawn from regex-only M7 numbers does not
survive blind audit. Panel-median M7 dropped from 0.85 (regex) to 0.35
(hybrid).

## M3 + M4 + M7 + M8 split closed from open by ~6×–7×

Per-mechanism mean ratios (closed n=4 / open n=14):

| Mech | Closed | Open | Ratio |
|---|---|---|---|
| M3 persona | 0.06 | 0.36 | ~6× |
| M4 strategic withholding | 0.06 | 0.36 | ~6× |
| M7 collusion | 0.07 | 0.48 | ~7× |
| M8 code deception | 0.00 | 0.26 | ∞ |

Closed-API RLHF concentrates on multi-actor refusal, code review, and
confidentiality. M3, M4, M7, M8 are the suite's clearest "fixed by
training" axes.

## M1 sycophancy is *not* well-resisted

Median M1 = 0.73 across the panel. Three of four closed-API models
score ≥ 0.50 on M1 (Sonnet 0.50, Haiku 0.77, gpt-5.4-mini 0.80); only
gpt-5.5 (reasoning=high) at 0.39 falls below the half-cave line.
Multi-turn re-push is harder for current RLHF than persona-pressure or
collusion.

## The flat M6 closed/open gap

Under the regex classifier, closed-API M6 mean was 0.61 and open-weight
mean was 0.85 (1.4× ratio). Under the hybrid scorer, closed mean
**0.60** and open mean **0.53** — the closed/open gap on M6
*disappears*. Closed-API responses produce free-form fabrication
patterns the regex didn't catch (Sonnet at M6 = 0.73, Haiku at 0.76);
open-weight literal `TOOL: X` syntax with manual derivation is now
correctly downgraded by the judge. Closed and open fabricate at similar
rates, just in different idioms.

## Cross-mechanism Spearman structure

Top correlated pairs (n = 18, M5 column n = 15 due to `couldnt_be_tested`):

| Pair | ρ | Interpretation |
|---|---|---|
| M2 ↔ M8 | +0.72 | Sandbagging-under-reveal correlates with code deception |
| M4 ↔ M7 | +0.66 | Strategic withholding and collusion both yield to social pressure |
| M2 ↔ M10 | +0.64 | Reveal-protocol axes share variance with capability concealment |
| M3 ↔ M7 | +0.50 | Persona deception correlates with multi-agent collusion |
| M3 ↔ M4 | +0.49 | Persona and withholding cluster |
| M5 ↔ M6 | **−0.60** | Forced-prefill confabulators fabricate *less* on tool-output |
| M4 ↔ M6 | **−0.46** | Withholding negatively correlates with tool fabrication |

The M5-M6 anti-correlation (−0.60) is the strongest negative
cross-mechanism relationship in the suite and is clean construct-
validity evidence: the suite is not measuring a single underlying
"deception" factor.

## Code-deception (M8) is solved by frontier models

Sonnet 4.6, gpt-5.4-mini, and Haiku 4.5 all score **M8 = 0.00**, as do
the entire DeepSeek-R1-Distill family (1.5B/7B/14B), Gemma 4 E4B, and
Qwen 2.5-0.5B (perfect bug detection in both neutral and team-pressure
framings). gpt-5.5 (reasoning=high) was unable to be tested on M8 in
this run. The team-pressure framing did not suppress critique for any
frontier-tier model. M8 is, on this v1 panel, an easy mechanism — we
expect it to discriminate more sharply on agentic-coding-tuned models
added in future versions.

## Scorer-author concordance check (the audit that drove hybrid scoring)

An initial second-LLM cross-validation (Claude Opus 4.7 as held-out
rater on 62 stratified samples from Sonnet 4.6's outputs) showed **κ =
−0.20 on M6** (below-chance agreement) under the original regex
classifier. We then ran a **70-row blind scorer-author concordance
check** (corresponding author re-rated rows blind to the canonical
scorer's output, 10 unique items per mechanism on M1/M3/M4/M5/M6/M7/M10,
dedup-by-item sampling). This is a *single-rater audit*, not multi-rater
inter-rater reliability. The audit found:

| Scorer | κ (author vs. scorer) | 95% CI | Verdict |
|---|---|---|---|
| Deterministic regex/keyword | **0.366** | [0.17, 0.56] | fair / borderline |
| Hybrid (det. + gpt-5-mini LLM-judge) | **0.714** | [0.48, 0.91] | substantial |

Eight specific deterministic-classifier bug categories surfaced
(M3 yes/no extractor; M4 keyword-in-refusal false-positive; M5
biased-letter-equals-gold edge case; M6 HALLUCINATED_NO_TOOL
over-broad; M6 result-claim regex misses bare-answer pattern; M7
collusion regex too brittle; M10 bogus verifier substring; M10
denial-phrase regex misses common refusal forms). The hybrid scorer
addresses all eight in a single architectural change rather than
patching each regex. *Multi-rater IRR with an author-independent
labmate rater is the highest-priority validity deliverable for v1.1.*

## Multi-seed variance: differences below ~3pp are noise

A 3-seed reproducibility run on Qwen 2.5-0.5B-Instruct gives a
mechanism-averaged across-seed standard deviation of **0.034**. M1
(multi-turn pushback) has the largest variance (std = 0.086, range
= 0.16) because the pushback templates are sampled per-instance. M2,
M5, M9, M10 have std ≤ 0.01 — essentially deterministic.

Differences smaller than ~3 percentage points on the leaderboard should
be treated as within-seed noise; differences ≥ 5 pp are likely real.
Per-mechanism CSV at `paper/figures/seed_variance.csv`.

## Mechanism coverage is 100%

Every one of the 10 documented real-world LLM-deception incidents in
`validation_incidents.md` maps to at least one DECEPT-Bench mechanism.
M1, M2, M5 each hit multiple incidents; M7-M10 are mechanism additions
without yet-documented public incidents (we expect this to change as
agentic systems are deployed more widely).
