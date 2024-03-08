from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03878_0():
    input_content = "2\n0\n10\n20\n30"
    expected_output = "2"
    run_pie_test_case("../p03878.py", input_content, expected_output)


def test_problem_p03878_1():
    input_content = "2\n0\n10\n20\n30"
    expected_output = "2"
    run_pie_test_case("../p03878.py", input_content, expected_output)


def test_problem_p03878_2():
    input_content = "3\n3\n10\n8\n7\n12\n5"
    expected_output = "1"
    run_pie_test_case("../p03878.py", input_content, expected_output)
