from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01420_0():
    input_content = "2 2 50\n30 50 1\n30 50 2"
    expected_output = "0.28770000\n0.71230000"
    run_pie_test_case("../p01420.py", input_content, expected_output)


def test_problem_p01420_1():
    input_content = "2 1 100\n100 100 10\n0 100 1"
    expected_output = "0.00000000\n1.00000000"
    run_pie_test_case("../p01420.py", input_content, expected_output)


def test_problem_p01420_2():
    input_content = "2 2 50\n30 0 1\n30 50 2"
    expected_output = "0.51000000\n0.49000000"
    run_pie_test_case("../p01420.py", input_content, expected_output)


def test_problem_p01420_3():
    input_content = "2 2 50\n30 50 1\n30 50 2"
    expected_output = "0.28770000\n0.71230000"
    run_pie_test_case("../p01420.py", input_content, expected_output)


def test_problem_p01420_4():
    input_content = "3 1 100\n50 1 1\n50 1 1\n50 1 1"
    expected_output = "0.12500000\n0.12500000\n0.12500000"
    run_pie_test_case("../p01420.py", input_content, expected_output)
