# Finer-timing experiment protocol

The current v2 model, submission, public sanity score and reports remain frozen. No competition labels are read by this experiment.

The first experiment keeps v2's trained classifier and transition matrix fixed. On J03WQQ only, compare original 5 Hz interpolation with native 25 Hz prediction, optionally average home/away exchanged predictions, and use evidence scales 1.0, 1.5 and 2.0 (12 configurations). Select by spellF1, macroF1 as tie-break. Stop if no development gain.

All seven IDSSE matches were already used in prior experiments. J03WOY and J03WR9 can support a regression check but must not be described as new untouched tests for this round. New corruption seeds on those same matches are robustness checks, not fresh matches.

Before any fresh cross-provider scores, audit public Metrica sample-event semantics. Use only provider-labelled intervals with unambiguous ownership/stoppages; do not invent a full frame-level possession ground truth from sparse events. If suitable, reserve sample games 1 and 2 as external confirmation, without tuning on their outcomes. Any sparse/event-defined diagnostic must be identified as such and must not be compared numerically with official full-possession spellF1.

Promotion requires a development improvement, no spellF1 regression on either previous external test, and confirmation on an independently sourced valid test. Otherwise retain v2 and report the experiment honestly. No public-window tuning, match-specific overrides or human corrections.

## Outcome

The frozen baseline won the 12-configuration comparison (one baseline plus eleven alternatives). Its development spellF1 was 0.614228; native 25 Hz prediction with the same decoder reached 0.609346. Native prediction slightly improved macroF1 from 0.904747 to 0.905245, illustrating why optimizing frame labels alone does not necessarily improve possession intervals. Home/away averaging and weaker sequence persistence also did not improve the ranking metric.

The experiment stopped at the predeclared development gate. No external test scores or competition public labels were read, no model was retrained, and the current submission was not replaced. The candidate CSV hash remains identical to the v2 release. Full results are in timing_development.json and outcome.json.

The IDSSE catalogue confirms seven matches total. Metrica public sample-event definitions were inspected for possible future cross-provider evaluation; they do not supply a directly interchangeable frame-level answer key. No sample matches were scored or used for training in this round.
