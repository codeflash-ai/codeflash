from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03164_0():
    input_content = "3 8\n3 30\n4 50\n5 60"
    expected_output = "90"
    run_pie_test_case("../p03164.py", input_content, expected_output)


def test_problem_p03164_1():
    input_content = "1 1000000000\n1000000000 10"
    expected_output = "10"
    run_pie_test_case("../p03164.py", input_content, expected_output)


def test_problem_p03164_2():
    input_content = "6 15\n6 5\n5 6\n6 4\n6 6\n3 5\n7 2"
    expected_output = "17"
    run_pie_test_case("../p03164.py", input_content, expected_output)


def test_problem_p03164_3():
    input_content = "3 8\n3 30\n4 50\n5 60"
    expected_output = "90"
    run_pie_test_case("../p03164.py", input_content, expected_output)
