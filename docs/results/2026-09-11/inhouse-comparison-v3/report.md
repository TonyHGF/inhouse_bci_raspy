# Aligned Inhouse fivefold comparison

Complete folds: 5/5; provisional=False

All models train directly on the retained outer 80%. No inner split, selection or refit exists in the fixed arms; their schedule is predetermined. Old preprocessing is session-wide before splitting; new preprocessing fits training data only.

The test-selected reference uses test feedback for scheduling, checkpoint selection and early stopping. Compare old_fixed with new_fixed to isolate preprocessing; neither fixed arm uses test feedback.

| Arm | Common trial accuracy | N |
|---|---:|---:|
| old_fixed | 68.51% | 235 |
| new_fixed | 62.98% | 235 |
| old_test_selected | 70.21% | 235 |
