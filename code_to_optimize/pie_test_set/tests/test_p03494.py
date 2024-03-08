from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03494_0():
    input_content = "3\n8 12 40"
    expected_output = "2"
    run_pie_test_case("../p03494.py", input_content, expected_output)


def test_problem_p03494_1():
    input_content = "6\n382253568 723152896 37802240 379425024 404894720 471526144"
    expected_output = "8"
    run_pie_test_case("../p03494.py", input_content, expected_output)


def test_problem_p03494_2():
    input_content = "4\n5 6 8 10"
    expected_output = "0"
    run_pie_test_case("../p03494.py", input_content, expected_output)


def test_problem_p03494_3():
    input_content = "3\n8 12 40"
    expected_output = "2"
    run_pie_test_case("../p03494.py", input_content, expected_output)
