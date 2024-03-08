from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02695_0():
    input_content = "3 4 3\n1 3 3 100\n1 2 2 10\n2 3 2 10"
    expected_output = "110"
    run_pie_test_case("../p02695.py", input_content, expected_output)


def test_problem_p02695_1():
    input_content = "3 4 3\n1 3 3 100\n1 2 2 10\n2 3 2 10"
    expected_output = "110"
    run_pie_test_case("../p02695.py", input_content, expected_output)


def test_problem_p02695_2():
    input_content = "10 10 1\n1 10 9 1"
    expected_output = "1"
    run_pie_test_case("../p02695.py", input_content, expected_output)


def test_problem_p02695_3():
    input_content = "4 6 10\n2 4 1 86568\n1 4 0 90629\n2 3 0 90310\n3 4 1 29211\n3 4 3 78537\n3 4 2 8580\n1 2 1 96263\n1 4 2 2156\n1 2 0 94325\n1 4 3 94328"
    expected_output = "357500"
    run_pie_test_case("../p02695.py", input_content, expected_output)
