# Academic Positioning For NeurIPS-Style Review

## Contribution Claim

The adapter is best framed as a **mechanism-aware applicability diagnostic** for
MetaGB, not as a supervised benchmark-win predictor.

It supports the third contribution of the broader MetaGB paper:

1. **Method**: feature-selected local linear leaves.
2. **System**: XGBoost hist integration.
3. **Analysis**: a theorem-guided preflight adapter that characterizes when the
   method's inductive bias is structurally applicable.

## Reviewer Concern 1: "Is This Just A Heuristic?"

Answer:

The score is derived from the same second-order objective used by MetaGB.
For a local region `R`, MetaGB's added basis direction has score:

```text
score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta)
```

The preflight score estimates whether this projected-gradient alignment exists
after a cheap conditional partition. The components are not arbitrary data
descriptors; each is tied to one MetaGB mechanism:

- local linear leaf advantage;
- conditional/predictive decoupling;
- local heterogeneity;
- predictive sparsity/concentration;
- local signal support.

## Reviewer Concern 2: "Why Not Ordinary Meta-Learning Or AutoML Landmarking?"

Answer:

The score does not train on historical benchmark outcomes. It is an intrinsic
mechanism score computed from the dataset and the MetaGB leaf objective. Outcome
labels are used only for retrospective audit.

This makes the adapter closer to an inductive-bias diagnostic than to a model
selection classifier.

## Reviewer Concern 3: "What If High Score Does Not Win?"

Answer:

A high score means the data contains MetaGB-compatible structure. If MetaGB does
not win, the failure is a mechanism-to-optimization conversion failure, not
necessarily evidence that the mechanism is absent.

Such cases should be analyzed for:

- noise;
- feature scale;
- binning;
- regularization;
- pred-feature sampling;
- hyperparameter selection;
- mismatch between training leaf statistics and prediction-time raw values.

## Reviewer Concern 4: "Is The Evidence Sufficient?"

Answer:

The current implementation creates the required evidence scaffolding:

1. synthetic mechanism recovery;
2. true predictive-feature hit rate;
3. real-data retrospective audit;
4. ablation;
5. sensitivity.

This is aligned with the broader reproducibility expectation in modern NeurIPS
research culture, where reliable code, clear evaluation protocols, and complete
experimental chains are important for trustworthiness. Pineau et al. (2020)
describe the NeurIPS reproducibility program and its emphasis on robust research
workflow. Király et al. (2018) similarly criticize incomplete empirical
argumentation in ML papers, motivating stronger baseline, comparison, and
uncertainty reporting.

## Current Evidence Status

Synthetic mechanism recovery is strong:

- piecewise-constant data scores low;
- local-linear and conditional/predictive decoupling data score high;
- high-dimensional sparse-noise data remains high;
- true predictive-feature hit rates are high in mechanism-aligned synthetic data.

Real-data audit is intentionally weaker and more nuanced:

- high mechanism score does not guarantee benchmark gain;
- high-score failures become case-analysis targets;
- real audit should not be used to fit the score.

## What Is Still Needed For A Strong Submission

Before claiming NeurIPS-level evidence, add:

- more synthetic families with graded noise and graded feature sparsity;
- confidence intervals for real benchmark audit;
- feature-hit analysis from actual trained MetaGB leaves where available;
- fixed-backbone pred-fraction sweeps by mechanism score band;
- real-data case analysis for high-score non-wins;
- comparison against generic dataset meta-features or AutoML-style landmarking
  as a contrast, while preserving the no-outcome-training claim.
