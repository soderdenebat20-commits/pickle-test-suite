# Pickle Stability & Correctness Test Suite

## Research Question
Does Python's pickle module produce byte-for-byte identical output for the same input across all data types and protocol versions (0–5)?

## Team Members
- Member 1: BAT-ERDENE SOD-ERDENE — TestEquivalencePartitioning, TestBoundaryValues, TestRoundTrip
- Member 2: AMINA AZAMAT — TestRecursiveStructures, TestProtocolVersions, TestFuzzing, TestWhiteBox

## How to Run
```bash
pip install pytest pytest-html
python -m pytest test_pickle_stability.py -v --html=test_results.html --self-contained-html
```

## Results
80/80 tests passed in ~0.16 seconds.

## Key Finding
set and frozenset pickle output may be non-deterministic due to Python's PYTHONHASHSEED hash randomization.# pickle-test-suite
