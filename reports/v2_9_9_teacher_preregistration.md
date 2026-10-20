# V2.9.9 | Archived teacher complement, pre-registered, no training

Registered before any V2.9.9 OOF retrieval or evaluation.

- Contest: playground-series-s6e10, exact 3-fold StratifiedKFold seed 20261018.
- Hypothesis: V2.9.4 independent original-data Teacher + route OOF may add a small orthogonal signal to the RealMLP and conditional-TE ensemble, despite failing to improve historical V2.9.3 by itself.
- Fixed 4-model rank fusion ONLY: V2.9.3 0.60, RealMLP V2.9.6 0.20, conditional TE V2.9.7 0.15, V2.9.4 Teacher-route rank OOF 0.05. No grid search and no weights selected after seeing the results.
- Reproduce official incumbent V2.9.6 CV 0.9609541991872741, V2.9.3 0.9607535043719558, V2.9.4 model 0.9607525307822926 before promotion.
- Check OOF rows (699635), test rows (299844), exact target order, canonical competition train/test IDs and independent new model ID alignment. V2.5, V2.8 and V2.9.4 archives do not store train IDs; only allow canonical-label/order audit, do not misrepresent these as explicit ID joins. Require all archived notebooks be the exact owner-preserved handles in the state files.
- Strict exploratory gate to limit multiple-testing effects: CV improvement >=0.00010 vs V2.9.6, at least two fold gains >0 and worst fold gain > -0.00005. Results stay candidate_ready_unsubmitted (never automatic Kaggle submission).
- Read-only Kaggle operations: retrieve existing notebooks, never push, rerun, create, modify, or submit a kernel. Do not persist or upload competition data or raw predictions. Store only metrics/artifact hashes.
- Experiments are exploratory on the same OOF; any winning model requires extra independent robustness review, Kaggle submission quota/dedupe guard, and an explicit pre-reservation process. Never claim OOF success is official Public LB.
- Execute only after V2.9.8 first produces terminal OOF evidence or when explicitly run after source audit; do not overlap other active OOF audit job.
