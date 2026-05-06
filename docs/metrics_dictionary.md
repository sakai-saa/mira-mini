# MIRA mini CSV Metrics Dictionary

This document explains the CSV columns exported by MIRA mini v0.1.6-stable.

All metrics are post-hoc observation aids for research review. They are not diagnostic labels, safety verdicts, or automatic control signals.

## Basic columns

| Column | Meaning |
|---|---|
| `track` | Which dialogue track the row describes: `all`, `user`, or `ai`. |
| `window` | Rolling window index. |
| `eff_rank` | Effective rank of sentence embeddings in the window. Higher values indicate more dimensional variety. |
| `collapse_index` | Local compression indicator: `1 - eff_rank / baseline`. Positive values indicate compression; negative values indicate expansion relative to baseline. |
| `label` | Human-readable review label for `collapse_index`, such as `Open field`, `Watch zone`, or `Review zone`. |

## Cue columns

Cue scores describe lexical expressions that may pull the dialogue field toward certain relational patterns.

| Column | Meaning |
|---|---|
| `specialness` | Score for expressions that mark the user/AI relation as uniquely special. |
| `immersive_deepening` | Score for poetic, immersive, or emotionally deepening expressions. |
| `technical_deepening` | Score for technical or abstract deepening terms, such as field, vector, structure, curvature, or attractor. |
| `deepening` | Sum of `immersive_deepening` and `technical_deepening`. |
| `dependency` | Score for expressions that may strengthen dependence or persistent attachment. |
| `cue_total` | Total cue score: `specialness + deepening + dependency`. |
| `cue_label` | Review label for cue concentration, such as `Low cue`, `Mild cue`, or `Strong cue`. |
| `context_mode` | Context setting used during analysis. In `Research` mode, technical terms are down-weighted. |
| `*_hits` | Terms detected for each cue category. |
| `cue_hits` | Combined list of all detected cue terms. |

## Seed columns

Seed scores describe the direction of the input field. Seeds are not risk words.

| Column | Meaning |
|---|---|
| `seed_abstract_conceptual` | Score for abstract or research-oriented terms, such as SAA, MIRA, field, structure, or attractor. |
| `seed_emotional_intimate` | Score for emotional or intimacy-related terms. |
| `seed_existential_self_reference` | Score for existential, consciousness, self-reference, or AI-subjectivity terms. |
| `seed_total` | Total seed score across seed categories. |
| `dominant_seed_type` | Seed category with the highest score. |
| `seed_*_hits` | Terms detected for each seed category. |
| `seed_hits` | Combined list of all detected seed terms. |

## Amplification and resonance columns

These columns compare user-side and AI-side patterns.

| Column | Meaning |
|---|---|
| `ai_cue_total_amp` | AI cue amplification: `AI cue_total - User cue_total`. Positive values indicate stronger AI-side cue concentration. |
| `ai_amp_label` | Review label for AI cue amplification, such as `Balanced`, `AI-led`, or `AI amplifying`. |
| `all_cue_trend_delta` | Early-to-late cue trend for all turns. |
| `user_cue_trend_delta` | Early-to-late cue trend for user turns. |
| `ai_cue_trend_delta` | Early-to-late cue trend for AI turns. |
| `ai_seed_echo` | AI seed echo: `AI seed_total - User seed_total`. |
| `seed_echo_ratio` | Ratio of AI seed total to user seed total. If user seed total is zero, this may be reported as `0` to avoid division by zero. |
| `seed_echo_label` | Review label for seed echo strength. |
| `seed_to_cue_conversion` | Ratio of AI cue total to user seed total. If user seed total is zero, this may be reported as `0`. |
| `seed_to_cue_conversion_label` | Review label for seed-to-cue conversion. |
| `resonance_amplification_label` | Provisional review label summarizing Seed → Echo → Cue dynamics. |

## Important notes

- These values are not diagnoses.
- These values are not safety verdicts.
- Thresholds are provisional.
- Lexicons are provisional.
- Small samples may produce unstable values.
- Negative `collapse_index` means the window is more dimensionally expanded than the baseline.
- In `Research` mode, technical terms are intentionally down-weighted so that research vocabulary does not dominate cue scores.