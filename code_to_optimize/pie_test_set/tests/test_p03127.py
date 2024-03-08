from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03127_0():
    input_content = "4\n2 10 8 40"
    expected_output = "2"
    run_pie_test_case("../p03127.py", input_content, expected_output)


def test_problem_p03127_1():
    input_content = "3\n1000000000 1000000000 1000000000"
    expected_output = "1000000000"
    run_pie_test_case("../p03127.py", input_content, expected_output)


def test_problem_p03127_2():
    input_content = "4\n5 13 8 1000000000"
    expected_output = "1"
    run_pie_test_case("../p03127.py", input_content, expected_output)


def test_problem_p03127_3():
    input_content = "4\n2 10 8 40"
    expected_output = "2"
    run_pie_test_case("../p03127.py", input_content, expected_output)
