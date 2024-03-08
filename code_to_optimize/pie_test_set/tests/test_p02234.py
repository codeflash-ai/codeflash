from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02234_0():
    input_content = "6\n30 35\n35 15\n15 5\n5 10\n10 20\n20 25"
    expected_output = "15125"
    run_pie_test_case("../p02234.py", input_content, expected_output)


def test_problem_p02234_1():
    input_content = "6\n30 35\n35 15\n15 5\n5 10\n10 20\n20 25"
    expected_output = "15125"
    run_pie_test_case("../p02234.py", input_content, expected_output)
