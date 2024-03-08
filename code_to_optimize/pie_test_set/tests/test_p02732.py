from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02732_0():
    input_content = "5\n1 1 2 1 2"
    expected_output = "2\n2\n3\n2\n3"
    run_pie_test_case("../p02732.py", input_content, expected_output)


def test_problem_p02732_1():
    input_content = "8\n1 2 1 4 2 1 4 1"
    expected_output = "5\n7\n5\n7\n7\n5\n7\n5"
    run_pie_test_case("../p02732.py", input_content, expected_output)


def test_problem_p02732_2():
    input_content = "5\n1 1 2 1 2"
    expected_output = "2\n2\n3\n2\n3"
    run_pie_test_case("../p02732.py", input_content, expected_output)


def test_problem_p02732_3():
    input_content = "4\n1 2 3 4"
    expected_output = "0\n0\n0\n0"
    run_pie_test_case("../p02732.py", input_content, expected_output)


def test_problem_p02732_4():
    input_content = "5\n3 3 3 3 3"
    expected_output = "6\n6\n6\n6\n6"
    run_pie_test_case("../p02732.py", input_content, expected_output)
