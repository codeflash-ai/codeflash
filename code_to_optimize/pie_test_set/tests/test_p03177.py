from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03177_0():
    input_content = "4 2\n0 1 0 0\n0 0 1 1\n0 0 0 1\n1 0 0 0"
    expected_output = "6"
    run_pie_test_case("../p03177.py", input_content, expected_output)


def test_problem_p03177_1():
    input_content = "1 1\n0"
    expected_output = "0"
    run_pie_test_case("../p03177.py", input_content, expected_output)


def test_problem_p03177_2():
    input_content = "4 2\n0 1 0 0\n0 0 1 1\n0 0 0 1\n1 0 0 0"
    expected_output = "6"
    run_pie_test_case("../p03177.py", input_content, expected_output)


def test_problem_p03177_3():
    input_content = "3 3\n0 1 0\n1 0 1\n0 0 0"
    expected_output = "3"
    run_pie_test_case("../p03177.py", input_content, expected_output)


def test_problem_p03177_4():
    input_content = "10 1000000000000000000\n0 0 1 1 0 0 0 1 1 0\n0 0 0 0 0 1 1 1 0 0\n0 1 0 0 0 1 0 1 0 1\n1 1 1 0 1 1 0 1 1 0\n0 1 1 1 0 1 0 1 1 1\n0 0 0 1 0 0 1 0 1 0\n0 0 0 1 1 0 0 1 0 1\n1 0 0 0 1 0 1 0 0 0\n0 0 0 0 0 1 0 0 0 0\n1 0 1 1 1 0 1 1 1 0"
    expected_output = "957538352"
    run_pie_test_case("../p03177.py", input_content, expected_output)


def test_problem_p03177_5():
    input_content = (
        "6 2\n0 0 0 0 0 0\n0 0 1 0 0 0\n0 0 0 0 0 0\n0 0 0 0 1 0\n0 0 0 0 0 1\n0 0 0 0 0 0"
    )
    expected_output = "1"
    run_pie_test_case("../p03177.py", input_content, expected_output)
