from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03662_0():
    input_content = "7\n3 6\n1 2\n3 1\n7 4\n5 7\n1 4"
    expected_output = "Fennec"
    run_pie_test_case("../p03662.py", input_content, expected_output)


def test_problem_p03662_1():
    input_content = "4\n1 4\n4 2\n2 3"
    expected_output = "Snuke"
    run_pie_test_case("../p03662.py", input_content, expected_output)


def test_problem_p03662_2():
    input_content = "7\n3 6\n1 2\n3 1\n7 4\n5 7\n1 4"
    expected_output = "Fennec"
    run_pie_test_case("../p03662.py", input_content, expected_output)
