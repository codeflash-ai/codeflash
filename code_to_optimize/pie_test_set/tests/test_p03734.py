from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03734_0():
    input_content = "4 6\n2 1\n3 4\n4 10\n3 4"
    expected_output = "11"
    run_pie_test_case("../p03734.py", input_content, expected_output)


def test_problem_p03734_1():
    input_content = "4 10\n1 100\n1 100\n1 100\n1 100"
    expected_output = "400"
    run_pie_test_case("../p03734.py", input_content, expected_output)


def test_problem_p03734_2():
    input_content = "4 1\n10 100\n10 100\n10 100\n10 100"
    expected_output = "0"
    run_pie_test_case("../p03734.py", input_content, expected_output)


def test_problem_p03734_3():
    input_content = "4 6\n2 1\n3 7\n4 10\n3 6"
    expected_output = "13"
    run_pie_test_case("../p03734.py", input_content, expected_output)


def test_problem_p03734_4():
    input_content = "4 6\n2 1\n3 4\n4 10\n3 4"
    expected_output = "11"
    run_pie_test_case("../p03734.py", input_content, expected_output)
