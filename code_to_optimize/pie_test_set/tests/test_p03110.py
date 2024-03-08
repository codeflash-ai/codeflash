from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03110_0():
    input_content = "2\n10000 JPY\n0.10000000 BTC"
    expected_output = "48000.0"
    run_pie_test_case("../p03110.py", input_content, expected_output)


def test_problem_p03110_1():
    input_content = "2\n10000 JPY\n0.10000000 BTC"
    expected_output = "48000.0"
    run_pie_test_case("../p03110.py", input_content, expected_output)


def test_problem_p03110_2():
    input_content = "3\n100000000 JPY\n100.00000000 BTC\n0.00000001 BTC"
    expected_output = "138000000.0038"
    run_pie_test_case("../p03110.py", input_content, expected_output)
