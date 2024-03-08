from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03449_0():
    input_content = "5\n3 2 2 4 1\n1 2 2 2 1"
    expected_output = "14"
    run_pie_test_case("../p03449.py", input_content, expected_output)


def test_problem_p03449_1():
    input_content = "7\n3 3 4 5 4 5 3\n5 3 4 4 2 3 2"
    expected_output = "29"
    run_pie_test_case("../p03449.py", input_content, expected_output)


def test_problem_p03449_2():
    input_content = "1\n2\n3"
    expected_output = "5"
    run_pie_test_case("../p03449.py", input_content, expected_output)


def test_problem_p03449_3():
    input_content = "5\n3 2 2 4 1\n1 2 2 2 1"
    expected_output = "14"
    run_pie_test_case("../p03449.py", input_content, expected_output)


def test_problem_p03449_4():
    input_content = "4\n1 1 1 1\n1 1 1 1"
    expected_output = "5"
    run_pie_test_case("../p03449.py", input_content, expected_output)
