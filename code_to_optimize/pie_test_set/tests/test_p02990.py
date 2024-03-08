from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02990_0():
    input_content = "5 3"
    expected_output = "3\n6\n1"
    run_pie_test_case("../p02990.py", input_content, expected_output)


def test_problem_p02990_1():
    input_content = "5 3"
    expected_output = "3\n6\n1"
    run_pie_test_case("../p02990.py", input_content, expected_output)


def test_problem_p02990_2():
    input_content = "2000 3"
    expected_output = "1998\n3990006\n327341989"
    run_pie_test_case("../p02990.py", input_content, expected_output)
