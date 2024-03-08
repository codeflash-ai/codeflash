from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03294_0():
    input_content = "3\n3 4 6"
    expected_output = "10"
    run_pie_test_case("../p03294.py", input_content, expected_output)


def test_problem_p03294_1():
    input_content = "5\n7 46 11 20 11"
    expected_output = "90"
    run_pie_test_case("../p03294.py", input_content, expected_output)


def test_problem_p03294_2():
    input_content = "7\n994 518 941 851 647 2 581"
    expected_output = "4527"
    run_pie_test_case("../p03294.py", input_content, expected_output)


def test_problem_p03294_3():
    input_content = "3\n3 4 6"
    expected_output = "10"
    run_pie_test_case("../p03294.py", input_content, expected_output)
