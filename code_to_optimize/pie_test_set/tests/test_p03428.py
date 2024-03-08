from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03428_0():
    input_content = "2\n0 0\n1 1"
    expected_output = "0.5\n0.5"
    run_pie_test_case("../p03428.py", input_content, expected_output)


def test_problem_p03428_1():
    input_content = "2\n0 0\n1 1"
    expected_output = "0.5\n0.5"
    run_pie_test_case("../p03428.py", input_content, expected_output)


def test_problem_p03428_2():
    input_content = "5\n0 0\n2 8\n4 5\n2 6\n3 10"
    expected_output = "0.43160120892732328768\n0.03480224363653196956\n0.13880483535586193855\n0.00000000000000000000\n0.39479171208028279727"
    run_pie_test_case("../p03428.py", input_content, expected_output)
