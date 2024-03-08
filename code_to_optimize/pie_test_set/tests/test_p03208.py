from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03208_0():
    input_content = "5 3\n10\n15\n11\n14\n12"
    expected_output = "2"
    run_pie_test_case("../p03208.py", input_content, expected_output)


def test_problem_p03208_1():
    input_content = "5 3\n10\n15\n11\n14\n12"
    expected_output = "2"
    run_pie_test_case("../p03208.py", input_content, expected_output)


def test_problem_p03208_2():
    input_content = "5 3\n5\n7\n5\n7\n7"
    expected_output = "0"
    run_pie_test_case("../p03208.py", input_content, expected_output)
