from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03184_0():
    input_content = "3 4 2\n2 2\n1 4"
    expected_output = "3"
    run_pie_test_case("../p03184.py", input_content, expected_output)


def test_problem_p03184_1():
    input_content = "5 2 2\n2 1\n4 2"
    expected_output = "0"
    run_pie_test_case("../p03184.py", input_content, expected_output)


def test_problem_p03184_2():
    input_content = "100000 100000 1\n50000 50000"
    expected_output = "123445622"
    run_pie_test_case("../p03184.py", input_content, expected_output)


def test_problem_p03184_3():
    input_content = "3 4 2\n2 2\n1 4"
    expected_output = "3"
    run_pie_test_case("../p03184.py", input_content, expected_output)


def test_problem_p03184_4():
    input_content = "5 5 4\n3 1\n3 5\n1 3\n5 3"
    expected_output = "24"
    run_pie_test_case("../p03184.py", input_content, expected_output)
