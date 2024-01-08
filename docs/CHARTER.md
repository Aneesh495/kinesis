# Kinesis: Project Charter

## 1. Mission and Scientific Scope
Kinesis investigates whether learned latent dynamics and epistemic uncertainty-aware planning improve closed-loop control under hidden changes in actuation and contact dynamics. In partially observed environments, policies often overfit to nominal dynamics and fail catastrophically when unmodeled delays, actuator deadbands, or friction shifts occur. Kinesis studies this problem by combining:
1. Stochastic recurrent state-space models (RSSM) that infer belief states over hidden physical latents.
2. Latent trajectory optimization (CEM and MPPI) evaluated directly in imagined latent space.
3. Epistemic uncertainty penalties derived from disagreement across world model ensembles to suppress over-optimistic plans in out-of-distribution regions.
4. Systematic evaluation across nominal dynamics and six frozen physical shift profiles in simulated continuous control benchmarks.

## 2. Research Questions
* Primary Question: Does epistemic uncertainty penalization in latent planning prevent policy degradation when contact and actuator parameters shift away from training distributions?
* Secondary Question: What is the empirical gap between one-step model prediction accuracy and multi-step closed-loop decision performance under partial observability?

## 3. Engineering Boundaries and Non-Goals
* Simulator Role: MuJoCo and the DeepMind Control Suite serve strictly as physical environment providers. They provide observations and execute actions. They are never cloned or stepped during online policy planning.
* Information Boundary: Policies observe only declared partial observations (delayed, noisy, or masked sensor values). Privileged simulator states, true friction values, actuator gains, and future observations are strictly hidden from policies during decision-making.
* Realism: Simulated results are reported honestly as simulation benchmarks and are not conflated with physical robot qualification.

## 4. Verification and Reproducibility Standards
* Independent Evaluator: All returns, violation costs, and success rates are independently audited from raw recorded transition streams.
* Negative Controls: Verifier suites include at least twelve negative controls to detect data leaks, altered configurations, and metric fabrication.
* Immutable Accounting: Replay buffers enforce checksums and episode boundaries to prevent temporal cross-contamination.
