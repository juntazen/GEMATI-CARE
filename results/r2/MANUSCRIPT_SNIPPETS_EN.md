# Ready-to-paste manuscript snippets (R2 additional analyses)

All numbers below appear in `r2_extra_metrics.json` or `r2_robustness_metrics.json`.
These analyses were added after the initial submission and evaluate the frozen seed-42 system;
selection used the validation set only.

## Section 3.2 — matched threshold, class-specific α, and McNemar (E1, E2, E4)

The reduction in missed crises relative to argmax was significant on paired data. At α = 0.05 the
router escalated 43 high-risk test messages that argmax missed and none that argmax escalated (exact
McNemar p = 2.3 × 10⁻¹³; at α = 0.10, 31 versus 0, p = 9.3 × 10⁻¹⁰). Because ESCALATE reduces to a
threshold on the high-risk probability, we also tuned that threshold on validation to the miss level of
the router at α = 0.05 (5.23%). The selected threshold, 0.096, is almost identical to the cut-off implied
by the conformal high-risk quantile (0.0951), and on the test set both rules escalated exactly the same
high-risk messages (miss 5.13%). The tuned threshold over-escalated fewer messages (35.00% versus 39.33%;
paired bootstrap difference −4.33 points, 95% CI −5.83 to −2.83) but under-escalated more (2.83% versus
2.17%; +0.67 points, CI 0.17 to 1.33), and expected cost did not differ (0.9642 versus 0.9583; difference
CI −0.0350 to 0.0542). The conformal layer therefore does not reduce error relative to a well-tuned
cut-off. Its contribution is to derive that cut-off from a stated miss budget using automatically labeled
calibration data, and to add the CLARIFY tier. Class-specific significance levels selected on validation
(α_low = 0.05, α_medium = 0.30, α_high = 0.05) gave a similar picture: over-escalation fell to 34.83%
(−4.50 points, CI −6.17 to −3.00) at an unchanged miss of 5.13%, under-escalation rose to 3.00% (+0.83
points, CI 0.17 to 1.67), and the number of escalations stayed at 413. The saving came from CLARIFY actions
that became SUPPORT, not from fewer human handovers.

## Section 3.2 (or 3.9) — recalibration on clinician labels (E3, R3)

To test the label-shift explanation directly, we refitted only the Mondrian quantiles on the
clinician-annotated validation set, keeping the classifier and temperature fixed, and evaluated on the
test set. With all 420 validation messages, low-risk coverage rose from 90.32% to 95.70% at α = 0.05
(exact binomial p = 0.87) and from 83.87% to 90.32% at α = 0.10 (p = 1.00), while high-risk coverage was
94.87% and 91.03%. In five-fold cross-conformal calibration (336 messages per fold), mean low-risk coverage
was 95.59% at α = 0.05 and 90.54% at α = 0.10, and no class fell significantly below its nominal level in
any fold. Medium-risk coverage instead rose above nominal (mean 99.22% and 97.00%). The low-risk shortfall
reported in Table 6 therefore disappears when calibration and evaluation labels come from the same
annotation process, which supports the exchangeability explanation. Clinician-label calibration increased
the review workload, however: at α = 0.05, over-escalation rose from 39.33% to 42.33% (cross-conformal
mean 43.07%) with the same 5.13% miss.

## Section 3.3 — five-seed stability (R1), one sentence

Across the five repeated splits, high-risk validation coverage averaged 93.14% ± 1.95% at α = 0.05 and
88.26% ± 1.19% at α = 0.10 and was not significantly below nominal in 18 of 20 seed × α cases, whereas
low- and medium-risk coverage fell significantly below nominal in 18 and 16 of the 20 cases.

## Table 11 — new row (TF-IDF benchmark, E5)

| Variant | Cold start | Median | p95 | Msg/s | Peak RSS |
|---|---|---|---|---|---|
| TF-IDF + logistic (primary) | 1.49 s | 2.35 ms | 5.61 ms | 670.14 | 432 MB |

Suggested text: "The primary TF-IDF classifier, measured under the same fresh-process protocol, started in
1.49 s, had a median latency of 2.35 ms (p95 5.61 ms), processed 670.14 messages per second, peaked at
432 MB of resident memory, and serialized to 6.24 MB. The timing covers the classifier; temperature
scaling, conformal sets, and routing add only a few arithmetic operations per message and were not timed
separately."

## Proposed Table 12 — matched threshold and class-specific α versus minimax routing (official test)

| Policy | Miss (95% CI) | Under | Over (95% CI) | Cost | S/C/E |
|---|---|---|---|---|---|
| Minimax α = 0.05 | 5.13% (2.64–7.98) | 2.17% | 39.33% (35.33–43.33) | 0.9583 | 33/154/413 |
| Matched threshold (t = 0.096) | 5.13% (2.64–7.98) | 2.83% | 35.00% (31.00–39.00) | 0.9642 | 65/124/411 |
| Class-specific α (0.05/0.30/0.05) | 5.13% (2.64–7.98) | 3.00% | 34.83% (30.83–38.67) | 0.9808 | 69/118/413 |
| Argmax | 23.50% (18.07–29.30) | 12.33% | 14.17% (11.50–17.33) | 1.2108 | 168/202/230 |

Caption: "Thresholds and α values were selected on validation; 95% CIs from 1,000 item-level bootstrap
replications on the test set."
