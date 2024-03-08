from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02409_0():
    input_content = "3\n1 1 3 8\n3 2 2 7\n4 3 8 1"
    expected_output = "0 0 8 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 7 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 1 0 0"
    run_pie_test_case("../p02409.py", input_content, expected_output)


def test_problem_p02409_1():
    input_content = "3\n1 1 3 8\n3 2 2 7\n4 3 8 1"
    expected_output = "0 0 8 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 7 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n####################\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 1 0 0"
    run_pie_test_case("../p02409.py", input_content, expected_output)
