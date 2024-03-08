from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03268_0():
    input_content = "3 2"
    expected_output = "9"
    run_pie_test_case("../p03268.py", input_content, expected_output)


def test_problem_p03268_1():
    input_content = "31415 9265"
    expected_output = "27"
    run_pie_test_case("../p03268.py", input_content, expected_output)


def test_problem_p03268_2():
    input_content = "3 2"
    expected_output = "9"
    run_pie_test_case("../p03268.py", input_content, expected_output)


def test_problem_p03268_3():
    input_content = "35897 932"
    expected_output = "114191"
    run_pie_test_case("../p03268.py", input_content, expected_output)


def test_problem_p03268_4():
    input_content = "5 3"
    expected_output = "1"
    run_pie_test_case("../p03268.py", input_content, expected_output)
