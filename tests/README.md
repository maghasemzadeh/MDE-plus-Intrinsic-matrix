# Test Suite for Depth Estimation Evaluation Pipeline

This test suite comprehensively validates all components of the depth estimation evaluation pipeline, from data loading to statistical analysis.

## Test Coverage

### ✅ All Tests Passing: 57/57

The test suite covers:

1. **Data Loading Tests** (`test_data_loading.py`)
   - PFM file reading (disparity maps)
   - Disparity to depth conversion
   - Calibration file parsing
   - Middlebury dataset item discovery and loading

2. **Error Calculation Tests** (`test_error_calculation.py`)
   - Basic error computation (|pred - gt|)
   - Error calculation with NaN values
   - Valid mask computation
   - Error statistics

3. **Metrics Calculation Tests** (`test_metrics.py`)
   - AbsRel (Absolute Relative Error) for metric models
   - RMSE (Root Mean Squared Error) for metric models
   - SILog (Scale-Invariant Log RMSE) for non-metric models
   - Edge cases: insufficient pixels, invalid values, zero/negative values

4. **Scale Factor Tests** (`test_scale_factor.py`)
   - Automatic scale factor calculation (median-based)
   - User-provided scale factors
   - Handling of invalid values

5. **Statistical Tests** (`test_statistical_tests.py`)
   - Welch's t-test (unequal variances)
   - Paired t-test
   - Bootstrap confidence intervals
   - Significance determination

6. **Integration Tests** (`test_integration.py`)
   - Full pipeline: data loading → depth computation → metrics
   - Two-camera comparison pipeline
   - Data consistency checks
   - Metrics consistency across cameras

## Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_metrics.py -v
```

### Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov=datasets --cov=models
```

### Run using the test runner:
```bash
python tests/run_tests.py
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_data_loading.py     # Data loading tests
├── test_error_calculation.py # Error computation tests
├── test_metrics.py          # Metrics calculation tests
├── test_scale_factor.py     # Scale factor tests
├── test_statistical_tests.py # Statistical analysis tests
├── test_integration.py      # Integration tests
├── run_tests.py             # Test runner script
└── README.md                # This file
```

## Key Test Scenarios

### Data Loading
- ✅ Reading PFM disparity files (big-endian and little-endian)
- ✅ Converting disparity to depth using calibration parameters
- ✅ Parsing Middlebury calibration files (single-line and multi-line matrices)
- ✅ Finding and loading dataset items

### Error Calculation
- ✅ Computing absolute error |pred - gt|
- ✅ Handling NaN and invalid pixels correctly
- ✅ Creating valid masks (finite, positive values)
- ✅ Computing error statistics

### Metrics
- ✅ AbsRel: mean(|pred - gt| / gt)
- ✅ RMSE: sqrt(mean((pred - gt)²))
- ✅ SILog: scale-invariant log RMSE with median alignment
- ✅ Edge cases: insufficient data, all invalid, zero/negative values

### Statistical Analysis
- ✅ Welch's t-test for comparing two independent samples
- ✅ Paired t-test for comparing related samples
- ✅ Bootstrap confidence intervals (95% and 99%)
- ✅ Significance determination (p < 0.05)

### Integration
- ✅ Complete pipeline from data loading to metrics
- ✅ Two-camera comparison workflow
- ✅ Data shape and type consistency
- ✅ Metrics consistency across cameras

## Test Results Summary

**Last Run:** All 57 tests passed ✅

- **Data Loading:** 10 tests
- **Error Calculation:** 8 tests
- **Metrics:** 10 tests
- **Scale Factor:** 7 tests
- **Statistical Tests:** 13 tests
- **Integration:** 9 tests

## Notes

- Tests use pytest fixtures for reusable test data
- Temporary directories are automatically cleaned up
- Tests validate both metric and non-metric model paths
- Edge cases are thoroughly tested (NaN, zero, negative values)
- Statistical tests verify correct implementation of t-tests and bootstrap CI

## Continuous Integration

These tests can be integrated into CI/CD pipelines to ensure code quality and correctness throughout development.

