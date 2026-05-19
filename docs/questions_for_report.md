# Questions to Address in the Final Academic Report

A running list of concepts, decisions, and plot interpretations from the project that need deeper explanation in the academic report.

Format: brief question → which section of the project it came from → priority (high / medium / low).

---

## Week 1 — Setup & Domain

- [ ] **What is a "piecewise-linear RUL target" mathematically?** (Domain notes / Week 2 modeling decisions) — high
- [ ] **Why is NASA's scoring function asymmetric? What's the derivation of the divisors 10 and 13?** (Domain notes Section 5) — medium
- [ ] **What is "walk-forward validation" and why does it matter for time-series?** (Domain notes Section 7) — high

## Week 2 — EDA

- [ ] **How exactly does `groupby + transform("max")` compute per-engine RUL?** (Task 4c code) — medium
- [ ] **Why does averaging across engines reveal trends that individual engines hide?** (Task 4c interpretation) — high
- [ ] **What does "single operating condition" actually mean physically for a turbofan?** (Q5) — medium
- [ ] **Why are sensors 9 and 14 "mixed" in individual view but clearly rising in averaged view?** (Task 4b vs 4c) — medium