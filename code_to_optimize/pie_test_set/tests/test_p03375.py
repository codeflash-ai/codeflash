from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03375_0():
    input_content = "2 1000000007"
    expected_output = "2"
    run_pie_test_case("../p03375.py", input_content, expected_output)


def test_problem_p03375_1():
    input_content = "3 1000000009"
    expected_output = "118"
    run_pie_test_case("../p03375.py", input_content, expected_output)


def test_problem_p03375_2():
    input_content = "2 1000000007"
    expected_output = "2"
    run_pie_test_case("../p03375.py", input_content, expected_output)


def test_problem_p03375_3():
    input_content = "3000 123456791"
    expected_output = "16369789"
    run_pie_test_case("../p03375.py", input_content, expected_output)


def test_problem_p03375_4():
    input_content = "50 111111113"
    expected_output = "1456748"
    run_pie_test_case("../p03375.py", input_content, expected_output)
