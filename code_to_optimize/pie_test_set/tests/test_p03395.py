from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03395_0():
    input_content = "3\n19 10 14\n0 3 4"
    expected_output = "160"
    run_pie_test_case("../p03395.py", input_content, expected_output)


def test_problem_p03395_1():
    input_content = "1\n50\n13"
    expected_output = "137438953472"
    run_pie_test_case("../p03395.py", input_content, expected_output)


def test_problem_p03395_2():
    input_content = "3\n19 15 14\n0 0 0"
    expected_output = "2"
    run_pie_test_case("../p03395.py", input_content, expected_output)


def test_problem_p03395_3():
    input_content = "3\n19 10 14\n0 3 4"
    expected_output = "160"
    run_pie_test_case("../p03395.py", input_content, expected_output)


def test_problem_p03395_4():
    input_content = "4\n2 0 1 8\n2 0 1 8"
    expected_output = "0"
    run_pie_test_case("../p03395.py", input_content, expected_output)


def test_problem_p03395_5():
    input_content = "2\n8 13\n5 13"
    expected_output = "-1"
    run_pie_test_case("../p03395.py", input_content, expected_output)
