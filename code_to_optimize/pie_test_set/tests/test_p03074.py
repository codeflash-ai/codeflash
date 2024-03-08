from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03074_0():
    input_content = "5 1\n00010"
    expected_output = "4"
    run_pie_test_case("../p03074.py", input_content, expected_output)


def test_problem_p03074_1():
    input_content = "5 1\n00010"
    expected_output = "4"
    run_pie_test_case("../p03074.py", input_content, expected_output)


def test_problem_p03074_2():
    input_content = "1 1\n1"
    expected_output = "1"
    run_pie_test_case("../p03074.py", input_content, expected_output)


def test_problem_p03074_3():
    input_content = "14 2\n11101010110011"
    expected_output = "8"
    run_pie_test_case("../p03074.py", input_content, expected_output)
