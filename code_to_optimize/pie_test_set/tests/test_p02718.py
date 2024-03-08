from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02718_0():
    input_content = "4 1\n5 4 2 1"
    expected_output = "Yes"
    run_pie_test_case("../p02718.py", input_content, expected_output)


def test_problem_p02718_1():
    input_content = "4 1\n5 4 2 1"
    expected_output = "Yes"
    run_pie_test_case("../p02718.py", input_content, expected_output)


def test_problem_p02718_2():
    input_content = "12 3\n4 56 78 901 2 345 67 890 123 45 6 789"
    expected_output = "Yes"
    run_pie_test_case("../p02718.py", input_content, expected_output)


def test_problem_p02718_3():
    input_content = "3 2\n380 19 1"
    expected_output = "No"
    run_pie_test_case("../p02718.py", input_content, expected_output)
