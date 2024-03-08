from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03003_0():
    input_content = "2 2\n1 3\n3 1"
    expected_output = "3"
    run_pie_test_case("../p03003.py", input_content, expected_output)


def test_problem_p03003_1():
    input_content = "10 9\n9 6 5 7 5 9 8 5 6 7\n8 6 8 5 5 7 9 9 7"
    expected_output = "191"
    run_pie_test_case("../p03003.py", input_content, expected_output)


def test_problem_p03003_2():
    input_content = (
        "20 20\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
    )
    expected_output = "846527861"
    run_pie_test_case("../p03003.py", input_content, expected_output)


def test_problem_p03003_3():
    input_content = "4 4\n3 4 5 6\n3 4 5 6"
    expected_output = "16"
    run_pie_test_case("../p03003.py", input_content, expected_output)


def test_problem_p03003_4():
    input_content = "2 2\n1 1\n1 1"
    expected_output = "6"
    run_pie_test_case("../p03003.py", input_content, expected_output)


def test_problem_p03003_5():
    input_content = "2 2\n1 3\n3 1"
    expected_output = "3"
    run_pie_test_case("../p03003.py", input_content, expected_output)
