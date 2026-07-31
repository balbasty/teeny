<!-- Title: conventional prefix + concise summary, e.g. "fix: …", "feat: …", "refactor: …" -->

## What & why

<!-- What this changes and the reason. Link the issue: Closes #NN  (or: part of #NN) -->

## How it was verified

- [ ] `make CXX=g++ run-test` and `make CXX=clang++ run-test` — all PASS
- [ ] new/affected `tests/test_*.cpp` added or updated
- [ ] ASan+UBSan clean on the affected host-path tests
- [ ] (slicing / broadcasting / layout folds) `make negindex` — the suite under `-DTNY_NO_NEGATIVE_INDEX` (CI runs it too)
- [ ] (core-path change) element-identity vs previous behaviour checked

## Notes

<!-- API/behaviour changes, follow-ups, anything a reviewer should know. -->
