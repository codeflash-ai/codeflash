from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03687_0():
    input_content = "serval"
    expected_output = "3"
    run_pie_test_case("../p03687.py", input_content, expected_output)


def test_problem_p03687_1():
    input_content = "serval"
    expected_output = "3"
    run_pie_test_case("../p03687.py", input_content, expected_output)


def test_problem_p03687_2():
    input_content = "whbrjpjyhsrywlqjxdbrbaomnw"
    expected_output = "8"
    run_pie_test_case("../p03687.py", input_content, expected_output)


def test_problem_p03687_3():
    input_content = "jackal"
    expected_output = "2"
    run_pie_test_case("../p03687.py", input_content, expected_output)


def test_problem_p03687_4():
    input_content = "zzz"
    expected_output = "0"
    run_pie_test_case("../p03687.py", input_content, expected_output)
