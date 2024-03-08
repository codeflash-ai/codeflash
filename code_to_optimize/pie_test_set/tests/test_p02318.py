from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02318_0():
    input_content = "acac\nacm"
    expected_output = "2"
    run_pie_test_case("../p02318.py", input_content, expected_output)


def test_problem_p02318_1():
    input_content = "icpc\nicpc"
    expected_output = "0"
    run_pie_test_case("../p02318.py", input_content, expected_output)


def test_problem_p02318_2():
    input_content = "acac\nacm"
    expected_output = "2"
    run_pie_test_case("../p02318.py", input_content, expected_output)
