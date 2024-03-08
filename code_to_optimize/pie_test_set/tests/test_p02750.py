from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02750_0():
    input_content = "3 7\n2 0\n3 2\n0 3"
    expected_output = "2"
    run_pie_test_case("../p02750.py", input_content, expected_output)


def test_problem_p02750_1():
    input_content = "1 3\n0 3"
    expected_output = "0"
    run_pie_test_case("../p02750.py", input_content, expected_output)


def test_problem_p02750_2():
    input_content = "5 21600\n2 14\n3 22\n1 3\n1 10\n1 9"
    expected_output = "5"
    run_pie_test_case("../p02750.py", input_content, expected_output)


def test_problem_p02750_3():
    input_content = "3 7\n2 0\n3 2\n0 3"
    expected_output = "2"
    run_pie_test_case("../p02750.py", input_content, expected_output)


def test_problem_p02750_4():
    input_content = "7 57\n0 25\n3 10\n2 4\n5 15\n3 22\n2 14\n1 15"
    expected_output = "3"
    run_pie_test_case("../p02750.py", input_content, expected_output)
