# MIRA mini

**MIRA mini** is a lightweight, non-diagnostic prototype for observing AI–user conversation dynamics.

It is designed as a small “conversation air meter” that visualizes aspects such as openness, loopiness, AI amplification, return paths, and the overall atmosphere of a short AI conversation.

MIRA mini is intended for personal, educational, and non-commercial research use.

---

## What MIRA mini is

MIRA mini is a simple local Streamlit application that allows a user to paste a short AI conversation and view lightweight indicators of the conversational field.

It focuses on the movement and structure of the conversation, rather than judging the user, the AI, or the content.

MIRA mini aims to provide a gentle way to observe:

- Openness
- Loopiness
- AI amplification
- Return path
- Conversation air
- Research metrics such as effective rank, compression index, seed echo ratio, seed-to-cue conversion, and cue amplification

---

## What MIRA mini is NOT

MIRA mini is **not**:

- a diagnostic tool
- a psychological assessment tool
- a medical tool
- a mental health evaluation tool
- an AI-dependency detector
- a safety classifier
- a risk-scoring system
- a tool for judging users, conversations, or AI systems

MIRA mini does not determine whether a conversation is “safe” or “dangerous.”  
It only provides lightweight, non-diagnostic indicators for observing conversational dynamics.

---

## Privacy and local use

MIRA mini is designed to run locally.

The prototype does not require:

- user registration
- login
- cloud storage
- database storage
- external user tracking

Users paste text into the local app, and the analysis is displayed locally.  
The project is designed around the principle of **no persistent storage of conversation logs**.

---

## Intended use

MIRA mini is intended for:

- personal exploration
- educational use
- non-commercial research
- prototyping conversational-field observation methods
- studying lightweight metrics for AI–user conversation dynamics

---

## Prohibited use

MIRA mini must not be used to:

- diagnose individuals
- evaluate mental health
- classify users as dependent, unsafe, risky, or vulnerable
- monitor people without consent
- score or rank users, conversations, or AI systems
- provide medical, psychological, legal, or safety-critical judgments
- deploy a commercial service based on the MIRA mini name, interface, or concept without permission

---

## Installation

Clone this repository and install the required packages:

```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run app_en.py
```
---

## Image asset


Place the main visual image at:

```text
assets/mira_main.png
```
This image is used as the main visual element in the app interface.

---

## Output files

MIRA mini can export research-oriented CSV files, including:

- conversation CSV
- measurement summary CSV

The summary CSV may include explanatory columns for each metric so that the output remains interpretable for research use.

---

## Version

Current prototype version:

**MIRA mini public v0.1.2 / v0.1.3-buttonstyle**

---

## v0.1.2 updates

- Removed incomplete HTML div structures that caused empty Streamlit cards to appear
- Fixed duplicated conversation log display in the research details section
- Added `st.session_state.last_result` to preserve measurement results
- Added `on_click="ignore"` to reduce UI state disruption during CSV download
- Separated conversation CSV and measurement summary CSV
- Added explanatory columns to the measurement summary CSV
- Improved metric text visibility

---

## v0.1.3-buttonstyle updates

- Changed CSV download button text color to match the MIRA mini brown color palette
- Changed CSV download button background to a white / cream tone
- Added a pink gradient hover effect for CSV download buttons
- Added a light hover effect for standard buttons

---

## License

The source code of MIRA mini is licensed under the PolyForm Noncommercial License 1.0.0.

MIRA mini is source-available for personal, educational, and non-commercial research use.

Commercial use, resale, SaaS deployment, paid integration, or redistribution as part of a commercial product is not permitted without prior written permission.

Please see `LICENSE` and `NOTICE.md` for details.

---

## Name and project identity

The names MIRA, MIRA mini, and MIRA Core refer to Jun Sakai’s MIRA project.

Modified versions of this software must clearly state that they are independent and not official MIRA mini releases.

The MIRA name, visual identity, or related project terminology must not be used for derivative tools that present themselves as:

- diagnostic tools
- AI-dependency assessment tools
- safety judgment tools
- psychological evaluation tools
- risk-scoring tools
- commercial monitoring services

without prior written permission.

---

## Citation

If you use MIRA mini in academic, educational, or non-commercial research contexts, please cite the project as follows:

Sakai, J. (2026). MIRA mini: A lightweight non-diagnostic prototype for observing AI–user conversation dynamics. Zenodo.

DOI: https://doi.org/10.5281/zenodo.20103803

---

## Disclaimer

MIRA mini is an experimental research prototype.

It is not intended to provide professional advice, diagnosis, safety evaluation, or psychological assessment.

The output should be interpreted only as lightweight indicators of conversational-field dynamics, not as judgments about users, AI systems, or conversations.

---

## Author

Jun Sakai
MIRA Project
2026
