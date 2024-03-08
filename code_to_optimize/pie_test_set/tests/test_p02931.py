from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02931_0():
    input_content = "6 2 2\n2 2 2\n1 1 8\n1 1 5\n1 2 9\n1 2 7\n2 1 4"
    expected_output = "28"
    run_pie_test_case("../p02931.py", input_content, expected_output)


def test_problem_p02931_1():
    input_content = "1 100000 100000\n1 1 1"
    expected_output = "1"
    run_pie_test_case("../p02931.py", input_content, expected_output)


def test_problem_p02931_2():
    input_content = "13 5 6\n1 3 35902\n4 6 19698\n4 6 73389\n3 6 3031\n3 1 4771\n1 4 4784\n2 1 36357\n2 1 24830\n5 6 50219\n4 6 22645\n1 2 30739\n1 4 68417\n1 5 78537"
    expected_output = "430590"
    run_pie_test_case("../p02931.py", input_content, expected_output)


def test_problem_p02931_3():
    input_content = "6 2 2\n2 2 2\n1 1 8\n1 1 5\n1 2 9\n1 2 7\n2 1 4"
    expected_output = "28"
    run_pie_test_case("../p02931.py", input_content, expected_output)
