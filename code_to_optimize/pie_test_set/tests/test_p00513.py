from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00513_0():
    input_content = "10\n4\n7\n9\n10\n12\n13\n16\n17\n19\n20"
    expected_output = "2"
    run_pie_test_case("../p00513.py", input_content, expected_output)


def test_problem_p00513_1():
    input_content = "None"
    expected_output = "None"
    run_pie_test_case("../p00513.py", input_content, expected_output)


def test_problem_p00513_2():
    input_content = "10\n4\n7\n9\n10\n12\n13\n16\n17\n19\n20"
    expected_output = "2"
    run_pie_test_case("../p00513.py", input_content, expected_output)
