from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02572_0():
    input_content = "3\n1 2 3"
    expected_output = "11"
    run_pie_test_case("../p02572.py", input_content, expected_output)


def test_problem_p02572_1():
    input_content = "3\n1 2 3"
    expected_output = "11"
    run_pie_test_case("../p02572.py", input_content, expected_output)


def test_problem_p02572_2():
    input_content = "4\n141421356 17320508 22360679 244949"
    expected_output = "437235829"
    run_pie_test_case("../p02572.py", input_content, expected_output)
