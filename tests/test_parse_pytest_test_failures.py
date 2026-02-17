from codeflash.verification.parse_test_output import parse_test_failures_from_stdout


def test_extracting_single_pytest_error_from_stdout():
    stdout = """
F...                                                                     [100%]
=================================== FAILURES ===================================
_______________________ test_calculate_portfolio_metrics _______________________

    def test_calculate_portfolio_metrics():
        # Test case 1: Basic portfolio
        investments = [
            ('Stocks', 0.6, 0.12),
            ('Bonds', 0.3, 0.04),
            ('Cash', 0.1, 0.01)
        ]
    
        result = calculate_portfolio_metrics(investments)
    
        # Check weighted return calculation
        expected_return = 0.6*0.12 + 0.3*0.04 + 0.1*0.01
        assert abs(result['weighted_return'] - expected_return) < 1e-10
    
        # Check volatility calculation
        expected_vol = math.sqrt((0.6*0.12)**2 + (0.3*0.04)**2 + (0.1*0.01)**2)
        assert abs(result['volatility'] - expected_vol) < 1e-10
    
        # Check Sharpe ratio
        expected_sharpe = (expected_return - 0.02) / expected_vol
>       assert abs(result['sharpe_ratio'] - expected_sharpe) < 1e-10
E       assert 4.109589046841222e-08 < 1e-10
E        +  where 4.109589046841222e-08 = abs((0.890411 - 0.8904109589041095))

code_to_optimize/tests/pytest/test_multiple_helpers.py:26: AssertionError
=========================== short test summary info ============================
FAILED code_to_optimize/tests/pytest/test_multiple_helpers.py::test_calculate_portfolio_metrics[ 1 ]
1 failed, 3 passed in 0.15s


"""
    errors = parse_test_failures_from_stdout(stdout)
    assert errors
    assert len(errors.keys()) == 1
    assert (
        errors["test_calculate_portfolio_metrics"]
        == """
    def test_calculate_portfolio_metrics():
        # Test case 1: Basic portfolio
        investments = [
            ('Stocks', 0.6, 0.12),
            ('Bonds', 0.3, 0.04),
            ('Cash', 0.1, 0.01)
        ]
    
        result = calculate_portfolio_metrics(investments)
    
        # Check weighted return calculation
        expected_return = 0.6*0.12 + 0.3*0.04 + 0.1*0.01
        assert abs(result['weighted_return'] - expected_return) < 1e-10
    
        # Check volatility calculation
        expected_vol = math.sqrt((0.6*0.12)**2 + (0.3*0.04)**2 + (0.1*0.01)**2)
        assert abs(result['volatility'] - expected_vol) < 1e-10
    
        # Check Sharpe ratio
        expected_sharpe = (expected_return - 0.02) / expected_vol
>       assert abs(result['sharpe_ratio'] - expected_sharpe) < 1e-10
E       assert 4.109589046841222e-08 < 1e-10
E        +  where 4.109589046841222e-08 = abs((0.890411 - 0.8904109589041095))

code_to_optimize/tests/pytest/test_multiple_helpers.py:26: AssertionError
"""
    )


def test_extracting_no_pytest_failures():
    stdout = """
....                                                                     [100%]
4 passed in 0.12s
"""
    errors = parse_test_failures_from_stdout(stdout)
    assert errors == {}


def test_extracting_multiple_pytest_failures_with_class_method():
    print("hi")

    stdout = """
F.F                                                                     [100%]
=================================== FAILURES ===================================
________________________ test_simple_failure ________________________

    def test_simple_failure():
        x = 1 + 1
>       assert x == 3
E       assert 2 == 3

code_to_optimize/tests/test_simple.py:10: AssertionError
________________ TestCalculator.test_divide_by_zero ________________

    class TestCalculator:
        def test_divide_by_zero(self):
>           Calculator().divide(10, 0)
E           ZeroDivisionError: division by zero

code_to_optimize/tests/test_calculator.py:22: ZeroDivisionError
=========================== short test summary info ============================
FAILED code_to_optimize/tests/test_simple.py::test_simple_failure
FAILED code_to_optimize/tests/test_calculator.py::TestCalculator::test_divide_by_zero
2 failed, 1 passed in 0.18s
"""
    errors = parse_test_failures_from_stdout(stdout)
    print(errors)
    assert len(errors) == 2

    assert "test_simple_failure" in errors
    assert (
        errors["test_simple_failure"]
        == """
    def test_simple_failure():
        x = 1 + 1
>       assert x == 3
E       assert 2 == 3

code_to_optimize/tests/test_simple.py:10: AssertionError
"""
    )

    assert "TestCalculator.test_divide_by_zero" in errors
    assert errors["TestCalculator.test_divide_by_zero"] == """
    class TestCalculator:
        def test_divide_by_zero(self):
>           Calculator().divide(10, 0)
E           ZeroDivisionError: division by zero

code_to_optimize/tests/test_calculator.py:22: ZeroDivisionError
"""


def test_extracting_from_invalid_pytest_stdout():
    stdout = """
Running tests...
Everything seems fine
No structured output here
Just some random logs
"""

    errors = parse_test_failures_from_stdout(stdout)
    assert errors == {}
