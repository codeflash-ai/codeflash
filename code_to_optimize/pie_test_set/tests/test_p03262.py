from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03262_0():
    input_content = "3 3\n1 7 11"
    expected_output = "2"
    run_pie_test_case("../p03262.py", input_content, expected_output)


def test_problem_p03262_1():
    input_content = "3 81\n33 105 57"
    expected_output = "24"
    run_pie_test_case("../p03262.py", input_content, expected_output)


def test_problem_p03262_2():
    input_content = "3 3\n1 7 11"
    expected_output = "2"
    run_pie_test_case("../p03262.py", input_content, expected_output)


def test_problem_p03262_3():
    input_content = "1 1\n1000000000"
    expected_output = "999999999"
    run_pie_test_case("../p03262.py", input_content, expected_output)
