from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03038_0():
    input_content = "3 2\n5 1 4\n2 3\n1 5"
    expected_output = "14"
    run_pie_test_case("../p03038.py", input_content, expected_output)


def test_problem_p03038_1():
    input_content = "10 3\n1 8 5 7 100 4 52 33 13 5\n3 10\n4 30\n1 4"
    expected_output = "338"
    run_pie_test_case("../p03038.py", input_content, expected_output)


def test_problem_p03038_2():
    input_content = "3 2\n5 1 4\n2 3\n1 5"
    expected_output = "14"
    run_pie_test_case("../p03038.py", input_content, expected_output)


def test_problem_p03038_3():
    input_content = "3 2\n100 100 100\n3 99\n3 99"
    expected_output = "300"
    run_pie_test_case("../p03038.py", input_content, expected_output)


def test_problem_p03038_4():
    input_content = "11 3\n1 1 1 1 1 1 1 1 1 1 1\n3 1000000000\n4 1000000000\n3 1000000000"
    expected_output = "10000000001"
    run_pie_test_case("../p03038.py", input_content, expected_output)
