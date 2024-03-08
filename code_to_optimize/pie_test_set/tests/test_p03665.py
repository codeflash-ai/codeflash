from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03665_0():
    input_content = "2 0\n1 3"
    expected_output = "2"
    run_pie_test_case("../p03665.py", input_content, expected_output)


def test_problem_p03665_1():
    input_content = "3 0\n1 1 1"
    expected_output = "4"
    run_pie_test_case("../p03665.py", input_content, expected_output)


def test_problem_p03665_2():
    input_content = "1 1\n50"
    expected_output = "0"
    run_pie_test_case("../p03665.py", input_content, expected_output)


def test_problem_p03665_3():
    input_content = "2 0\n1 3"
    expected_output = "2"
    run_pie_test_case("../p03665.py", input_content, expected_output)


def test_problem_p03665_4():
    input_content = "45 1\n17 55 85 55 74 20 90 67 40 70 39 89 91 50 16 24 14 43 24 66 25 9 89 71 41 16 53 13 61 15 85 72 62 67 42 26 36 66 4 87 59 91 4 25 26"
    expected_output = "17592186044416"
    run_pie_test_case("../p03665.py", input_content, expected_output)
