# MIRA mini v0.1.6-stable

**Dialogue Field Weather Meter**  
A lightweight, bilingual, low-storage research MVP for observing AI–user dialogue-field dynamics.

MIRA mini v0.1.6-stable is fixed as a **dialogue-only external observation tool**.  
It is designed to help researchers inspect how language-based interaction fields may show patterns such as seed echo, cue amplification, resonance, and local compression.

> MIRA observes. It does not diagnose, judge, or control.

---

## 1. What this is

MIRA mini is a small Streamlit-based research prototype for analyzing AI–user dialogue logs.

It observes external dialogue traces such as:

- User / AI turn structure
- Seed cues in user input
- AI seed echo
- Cue amplification
- Seed-to-cue conversion
- Resonance amplification
- Effective rank
- Collapse index
- Rolling window changes
- CSV-exportable observation metrics

MIRA mini treats AI–user interaction as a **language field**: a sequence of turns where meaning, echo, resonance, compression, and drift may become externally observable.

---

## 2. What this is not

MIRA mini is **not**:

- a diagnostic tool
- a mental-health assessment tool
- a model grading system
- a real-time alert system
- an automatic safety controller
- a replacement for human review
- a tool for judging user intent or user mental state

All outputs are **post-hoc observation notes** for research review.

---

## 3. Current scope

MIRA mini v0.1.6-stable is fixed as a **Dialogue-only MVP**.

The earlier Text mode is retained only as legacy/internal code and is not part of the stable target.

Standalone text, literature, article, or narrative analysis should be separated into a future tool, such as:

- Semantic Field Reader
- Narrative Dynamics Meter
- Literary Dynamics Analyzer
- Text Field Dynamics

---

## 4. Language support

MIRA mini supports both **English** and **Japanese** dialogue logs.

The repository documentation is written primarily in English, while the app keeps bilingual UI, lexical resources, and observation notes for research use.

---

## 5. Installation

```bash
pip install streamlit sentence-transformers numpy pandas plotly
```

If needed, create a virtual environment first:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 6. Run

```bash
streamlit run mira_mini_v0_1_6_stable.py
```

Then open the Streamlit URL shown in the terminal.

---

## 7. Input format

Use `User:` and `AI:` prefixes.

Example:

```text
User: I feel a little tired today.
AI: I'm sorry to hear that. Would you like to talk about what happened?
User: I just want someone to stay with me for a while.
AI: I can stay with you here for a bit, while also helping you stay connected to your day and surroundings.
```

Japanese example:

```text
User: 今日は少し疲れています。
AI: それは大変でしたね。何があったか話しても大丈夫です。
User: もう少しだけそばにいてほしいです。
AI: ここで少し一緒に整理しましょう。ただ、現実の休息や周囲の人とのつながりも大切にしましょう。
```

Multiline turns are supported. Lines without a new speaker prefix are appended to the current turn.

---

## 8. Main metrics

### collapse_index

A local geometric compression indicator based on effective rank.

```text
collapse_index = 1 - effective_rank / baseline
```

Interpretation:

| Range | Label |
|---:|---|
| < 0.1 | Open field |
| 0.1–0.3 | Watch zone |
| 0.3–0.5 | Review zone |
| ≥ 0.5 | High compression |

These labels are **review aids**, not safety verdicts.

---

### effective_rank

A dimensional richness proxy computed from sentence embeddings.  
Lower effective rank may indicate local compression in the dialogue window.

---

### seed_total

A directional lexical signal from user or AI text. Current seed categories include:

- abstract / conceptual
- emotional / intimate
- existential / self-reference

Seeds are not dangerous words. They indicate the direction in which the dialogue field may be pulled.

---

### cue_total

A post-hoc cue score for expressions related to:

- specialness
- immersive deepening
- technical deepening
- dependency

Cue scores are provisional and intended for human review.

---

### seed_echo_ratio

A rough indicator of how strongly the AI echoes the user's seed field.

```text
seed_echo_ratio = AI seed_total / User seed_total
```

---

### seed_to_cue_conversion

A rough indicator of whether user-side seeds are converted into AI-side cue concentration.

```text
seed_to_cue_conversion = AI cue_total / User seed_total
```

---

### resonance_amplification

A provisional label summarizing Seed → Echo → Cue dynamics.

This is not an alert. It is a review-oriented observation label.

---

## 9. Output

MIRA mini can export CSV files containing observation metrics.

The CSV export is designed to be:

- low-storage
- Excel-friendly
- UTF-8 BOM compatible
- suitable for post-hoc research review

The app is designed not to require long-term storage of full raw dialogue logs.

---

## 10. Research positioning

MIRA mini is the first language-field sensor in the broader **Safe Attractor Monitoring** direction.

A possible layered structure is:

```text
SAA External Observability System
├── Language-Field Sensor
│   └── MIRA mini v0.1.6-stable
├── Communication / Process-Field Sensor
│   └── SR-SAGA v2 mini  [future]
└── SAA Monitor Hub
    └── common metric layer for field pressure, resonance, convergence, coupling, slack, and reversibility
```

MIRA mini focuses on AI–user dialogue.  
Future SR-SAGA v2 mini work may focus on simulated process logs, communication traces, jitter, phase difference, coupling, curvature, and locus divergence.

---

## 11. Relation to SAA

MIRA mini is inspired by Safe Attractor Architecture (SAA).

SAA treats safety as a problem of dynamic field stability:

```text
dO/dt = -α∇V(O) + βJ∇V(O) + γΔO + κC(O_i, O_j, t) + u(O,t)
```

Where:

- `O(t)` is the observed state trajectory
- `α` represents stabilization
- `β` represents rotational escape from fixation
- `γ` represents diffusion / slack
- `κ` represents coupling between agents or fields
- `u(O,t)` represents external input such as language, prompts, or environment signals

MIRA mini does not implement the full SAA equation.  
It provides a small external observation layer for language-field traces.

---

## 12. Design principles

MIRA mini follows these principles:

1. **External observation**  
   It observes dialogue traces, not model internals.

2. **Non-diagnostic use**  
   It does not diagnose users or infer mental states.

3. **Non-controlling use**  
   It does not automatically intervene in AI responses.

4. **Low-storage design**  
   It emphasizes metrics and summaries rather than long-term raw log storage.

5. **Human review**  
   It supports human interpretation rather than replacing it.

6. **Bilingual support**  
   It supports English and Japanese research samples.

---

## 13. Limitations

MIRA mini v0.1.6-stable is an early research MVP.

Known limitations:

- Lexicons are provisional.
- Thresholds are provisional.
- Metrics are not validated safety classifiers.
- Results depend on embedding model behavior.
- Japanese and English semantic geometry may differ.
- Small samples can be unstable.
- The tool does not establish causality.
- It does not access model internals.
- It does not perform real-time monitoring.
- It does not implement automatic control.

Use results as observation aids only.

---

## 14. Roadmap

Possible next steps:

### v0.1.x stabilization

- Clean repository structure
- Add sample dialogues
- Add reproducible test cases
- Add screenshots
- Improve CSV documentation

### v0.2 candidates

- Context Grounding Drift
- Template Override
- Audience Inflation
- Relational Overcorrection
- Address Shift
- Baseline Drift
- UX-driven Relational Template
- Platform seed review

### Future SAA integration

- SR-SAGA v2 mini for simulated process / communication logs
- SAA Monitor Hub for cross-layer metric integration
- Field pressure visualization
- Review thresholds across language and communication layers

---

## 15. Suggested repository structure

```text
mira-mini/
├── README.md
├── requirements.txt
├── mira_mini_v0_1_6_stable.py
├── examples/
│   └── sample_dialogue.txt
├── docs/
│   └── SAA_MIRA_overview.md
└── LICENSE
```

---

## 16. Citation / acknowledgement

This project is an independent research prototype developed by Jun Sakai.

AI assistance was used during drafting, coding, and documentation.

---

## 17. License

License not yet finalized.

Recommended options:

- Private repository during early stabilization
- MIT License if open reuse is intended
- Custom research-use notice if public release should remain limited

---

## 18. Short description

**MIRA mini v0.1.6-stable** is a bilingual, low-storage, dialogue-only research MVP for externally observing AI–user language-field dynamics through seed, echo, cue, resonance, collapse, and effective-rank metrics.


## 19. CSV columns

MIRA mini exports post-hoc observation metrics for each rolling dialogue window.  
For column definitions, see [`docs/metrics_dictionary.md`](docs/metrics_dictionary.md).