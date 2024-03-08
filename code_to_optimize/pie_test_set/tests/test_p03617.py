from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03617_0():
    input_content = "20 30 70 90\n3"
    expected_output = "150"
    run_pie_test_case("../p03617.py", input_content, expected_output)


def test_problem_p03617_1():
    input_content = "10 100 1000 10000\n1"
    expected_output = "40"
    run_pie_test_case("../p03617.py", input_content, expected_output)


def test_problem_p03617_2():
    input_content = "20 30 70 90\n3"
    expected_output = "150"
    run_pie_test_case("../p03617.py", input_content, expected_output)


def test_problem_p03617_3():
    input_content = "10000 1000 100 10\n1"
    expected_output = "100"
    run_pie_test_case("../p03617.py", input_content, expected_output)


def test_problem_p03617_4():
    input_content = "12345678 87654321 12345678 87654321\n123456789"
    expected_output = "1524157763907942"
    run_pie_test_case("../p03617.py", input_content, expected_output)
