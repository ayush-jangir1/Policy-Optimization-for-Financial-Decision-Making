**# Policy-Optimization-for-Financial-Decision-Making**

End-to-end project using deep learning and offline reinforcement learning to optimize loan approvals on the LendingClub dataset (2007–2018).
We (1) build a supervised default-risk model, (2) frame the problem as offline RL with approve/deny actions, and (3) compare policies with business-focused metrics.

**What’s inside**

Task 1 – EDA & Preprocessing: feature selection, cleaning, and a reusable preprocessor.joblib.

Task 2 – Supervised DL: MLP classifier to predict default probability, AUC/F1 + profit curve to pick an approval threshold.

Task 3 – Offline RL: Conservative Q-Learning (CQL) learns an approve/deny policy from static logs.

Task 4 – Analysis & Comparison: DL threshold policy vs RL policy, OPE (on-support, IPS, DM, DR), slice analyses, and recommendations.

**How to run**

1) Task 1 — EDA & Preprocessing

Open notebooks/01_eda_and_preprocessing.ipynb.

Run all cells to generate artifacts/preprocessor.joblib.

2) Task 2 — Supervised default risk

Open notebooks/02_supervised_default_risk.ipynb.

Trains an MLP classifier and saves artifacts/dl_best.pt.

Produces AUC/F1 and a profit curve to select a decision threshold.

3) Tasks 3 & 4 — Offline RL + Analysis

Open notebooks/03_&_04_combined.ipynb.

Builds an offline, one-step bandit dataset (approve=1, deny=0).

Trains CQL and saves policy to artifacts/rl_policy/.

Computes OPE: on-support, IPS (clipped), Direct Method, Doubly Robust.




**Reward Design**

Current reward:

Approve & Fully Paid: + loan_amount × interest_rate

Approve & Default: − loan_amount

Deny: 0

This asymmetric scale emphasizes downside risk; it is conservative by design.

For deployment-grade evaluation, refine to:

Scale interest by term years (× term_months/12)

Add recovery rate on default (e.g., 40%)

Optionally include servicing costs, discounting, late fees

Re-train CQL and re-run OPE after changing rewards to get business-comparable value.


**Limitations & next steps**

Reward realism: incorporate term and recovery to align with P&L.

Deny state coverage: rejected loans used placeholders; add richer deny-side features if available.

Add Fitted Q Evaluation (FQE) for stronger OPE beyond IPS/DM/DR.

Compare additional algorithms: IQL, TD3+BC, and VW (contextual bandits).

Exports slice tables and comparison results to analysis_artifacts/.

Use these outputs to finalize reports/final_report.pdf.
