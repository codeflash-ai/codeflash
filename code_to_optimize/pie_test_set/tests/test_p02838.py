from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02838_0():
    input_content = "3\n1 2 3"
    expected_output = "6"
    run_pie_test_case("../p02838.py", input_content, expected_output)


def test_problem_p02838_1():
    input_content = "3\n1 2 3"
    expected_output = "6"
    run_pie_test_case("../p02838.py", input_content, expected_output)


def test_problem_p02838_2():
    input_content = "10\n3 14 159 2653 58979 323846 2643383 27950288 419716939 9375105820"
    expected_output = "103715602"
    run_pie_test_case("../p02838.py", input_content, expected_output)


def test_problem_p02838_3():
    input_content = "10\n3 1 4 1 5 9 2 6 5 3"
    expected_output = "237"
    run_pie_test_case("../p02838.py", input_content, expected_output)
