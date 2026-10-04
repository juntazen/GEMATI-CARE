# R2 extra experiments — summary

Sanity (must equal Table 5, alpha=0.05): miss=0.0513, over=0.3933

## E1 matched threshold
threshold=0.096  test: {'action_accuracy': 0.621667, 'under_escalation_rate': 0.028333, 'over_escalation_rate': 0.35, 'high_risk_miss_rate': 0.051282, 'expected_cost': 0.964167, 'action_distribution': {'SUPPORT': 65, 'CLARIFY': 124, 'ESCALATE': 411}}

## E2 class-specific alpha (selected on validation)
alpha_high=0.05: alphas={'low': 0.05, 'medium': 0.3, 'high': 0.05} test={'action_accuracy': 0.621667, 'under_escalation_rate': 0.03, 'over_escalation_rate': 0.348333, 'high_risk_miss_rate': 0.051282, 'expected_cost': 0.980833, 'action_distribution': {'SUPPORT': 69, 'CLARIFY': 118, 'ESCALATE': 413}}
alpha_high=0.1: alphas={'low': 0.05, 'medium': 0.05, 'high': 0.1} test={'action_accuracy': 0.658333, 'under_escalation_rate': 0.041667, 'over_escalation_rate': 0.3, 'high_risk_miss_rate': 0.102564, 'expected_cost': 0.904167, 'action_distribution': {'SUPPORT': 61, 'CLARIFY': 215, 'ESCALATE': 324}}

## E3 recalibration on clinician labels
alpha=0.05 auto-labels: {"low": 0.9032, "medium": 0.9611, "high": 0.9487, "overall": 0.9383}
alpha=0.05 clinician 420: {"low": 0.957, "medium": 0.9944, "high": 0.9487, "overall": 0.965}
alpha=0.1 auto-labels: {"low": 0.8387, "medium": 0.9111, "high": 0.8974, "overall": 0.8833}
alpha=0.1 clinician 420: {"low": 0.9032, "medium": 0.9722, "high": 0.9103, "overall": 0.9267}

## E4 McNemar
{
 "minimax_0.05_vs_argmax": {
  "escalated_by_first_only": 43,
  "escalated_by_second_only": 0,
  "discordant": 43,
  "p_exact": 2.2737367544323206e-13
 },
 "minimax_0.10_vs_argmax": {
  "escalated_by_first_only": 31,
  "escalated_by_second_only": 0,
  "discordant": 31,
  "p_exact": 9.313225746154785e-10
 },
 "minimax_0.05_vs_matched_threshold": {
  "escalated_by_first_only": 0,
  "escalated_by_second_only": 0,
  "discordant": 0,
  "p_exact": 1.0
 },
 "minimax_0.10_vs_expected_cost": {
  "escalated_by_first_only": 1,
  "escalated_by_second_only": 4,
  "discordant": 5,
  "p_exact": 0.375
 }
}

## E5 TF-IDF CPU benchmark
{
 "cold_start_s": 1.4851709743961692,
 "warm_single_ms_p50": 2.350918482989073,
 "warm_single_ms_p95": 5.607872549444434,
 "throughput_msgs_per_s_mean": 670.1359085479327,
 "peak_rss_mb": 432.2265625,
 "serialized_model_mb": 6.242494
}
