from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03774_0():
    input_content = "2 2\n2 0\n0 0\n-1 0\n1 0"
    expected_output = "2\n1"
    run_pie_test_case("../p03774.py", input_content, expected_output)


def test_problem_p03774_1():
    input_content = "3 4\n10 10\n-10 -10\n3 3\n1 2\n2 3\n3 5\n3 5"
    expected_output = "3\n1\n2"
    run_pie_test_case("../p03774.py", input_content, expected_output)


def test_problem_p03774_2():
    input_content = "2 2\n2 0\n0 0\n-1 0\n1 0"
    expected_output = "2\n1"
    run_pie_test_case("../p03774.py", input_content, expected_output)


def test_problem_p03774_3():
    input_content = "5 5\n-100000000 -100000000\n-100000000 100000000\n100000000 -100000000\n100000000 100000000\n0 0\n0 0\n100000000 100000000\n100000000 -100000000\n-100000000 100000000\n-100000000 -100000000"
    expected_output = "5\n4\n3\n2\n1"
    run_pie_test_case("../p03774.py", input_content, expected_output)
