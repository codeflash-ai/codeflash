from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03173_0():
    input_content = "4\n10 20 30 40"
    expected_output = "190"
    run_pie_test_case("../p03173.py", input_content, expected_output)


def test_problem_p03173_1():
    input_content = "6\n7 6 8 6 1 1"
    expected_output = "68"
    run_pie_test_case("../p03173.py", input_content, expected_output)


def test_problem_p03173_2():
    input_content = "5\n10 10 10 10 10"
    expected_output = "120"
    run_pie_test_case("../p03173.py", input_content, expected_output)


def test_problem_p03173_3():
    input_content = "3\n1000000000 1000000000 1000000000"
    expected_output = "5000000000"
    run_pie_test_case("../p03173.py", input_content, expected_output)


def test_problem_p03173_4():
    input_content = "4\n10 20 30 40"
    expected_output = "190"
    run_pie_test_case("../p03173.py", input_content, expected_output)
