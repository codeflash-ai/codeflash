from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02795_0():
    input_content = "3\n7\n10"
    expected_output = "2"
    run_pie_test_case("../p02795.py", input_content, expected_output)


def test_problem_p02795_1():
    input_content = "3\n7\n10"
    expected_output = "2"
    run_pie_test_case("../p02795.py", input_content, expected_output)


def test_problem_p02795_2():
    input_content = "2\n100\n200"
    expected_output = "2"
    run_pie_test_case("../p02795.py", input_content, expected_output)


def test_problem_p02795_3():
    input_content = "14\n12\n112"
    expected_output = "8"
    run_pie_test_case("../p02795.py", input_content, expected_output)
