from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03207_0():
    input_content = "3\n4980\n7980\n6980"
    expected_output = "15950"
    run_pie_test_case("../p03207.py", input_content, expected_output)


def test_problem_p03207_1():
    input_content = "4\n4320\n4320\n4320\n4320"
    expected_output = "15120"
    run_pie_test_case("../p03207.py", input_content, expected_output)


def test_problem_p03207_2():
    input_content = "3\n4980\n7980\n6980"
    expected_output = "15950"
    run_pie_test_case("../p03207.py", input_content, expected_output)
