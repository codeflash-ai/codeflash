from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03721_0():
    input_content = "3 4\n1 1\n2 2\n3 3"
    expected_output = "3"
    run_pie_test_case("../p03721.py", input_content, expected_output)


def test_problem_p03721_1():
    input_content = "3 4\n1 1\n2 2\n3 3"
    expected_output = "3"
    run_pie_test_case("../p03721.py", input_content, expected_output)


def test_problem_p03721_2():
    input_content = "10 500000\n1 100000\n1 100000\n1 100000\n1 100000\n1 100000\n100000 100000\n100000 100000\n100000 100000\n100000 100000\n100000 100000"
    expected_output = "1"
    run_pie_test_case("../p03721.py", input_content, expected_output)
