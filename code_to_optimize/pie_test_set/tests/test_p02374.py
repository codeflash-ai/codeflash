from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02374_0():
    input_content = "6\n2 1 2\n2 3 5\n0\n0\n0\n1 4\n7\n1 1\n0 3 10\n1 2\n0 4 20\n1 3\n0 5 40\n1 4"
    expected_output = "0\n0\n10\n60"
    run_pie_test_case("../p02374.py", input_content, expected_output)


def test_problem_p02374_1():
    input_content = "6\n2 1 2\n2 3 5\n0\n0\n0\n1 4\n7\n1 1\n0 3 10\n1 2\n0 4 20\n1 3\n0 5 40\n1 4"
    expected_output = "0\n0\n10\n60"
    run_pie_test_case("../p02374.py", input_content, expected_output)


def test_problem_p02374_2():
    input_content = "2\n1 1\n0\n4\n0 1 1\n1 1\n0 1 1\n1 1"
    expected_output = "1\n2"
    run_pie_test_case("../p02374.py", input_content, expected_output)


def test_problem_p02374_3():
    input_content = "4\n1 1\n1 2\n1 3\n0\n6\n0 3 1000\n0 2 1000\n0 1 1000\n1 1\n1 2\n1 3"
    expected_output = "1000\n2000\n3000"
    run_pie_test_case("../p02374.py", input_content, expected_output)
