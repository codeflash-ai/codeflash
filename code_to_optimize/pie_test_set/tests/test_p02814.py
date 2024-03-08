from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02814_0():
    input_content = "2 50\n6 10"
    expected_output = "2"
    run_pie_test_case("../p02814.py", input_content, expected_output)


def test_problem_p02814_1():
    input_content = "3 100\n14 22 40"
    expected_output = "0"
    run_pie_test_case("../p02814.py", input_content, expected_output)


def test_problem_p02814_2():
    input_content = "2 50\n6 10"
    expected_output = "2"
    run_pie_test_case("../p02814.py", input_content, expected_output)


def test_problem_p02814_3():
    input_content = "5 1000000000\n6 6 2 6 2"
    expected_output = "166666667"
    run_pie_test_case("../p02814.py", input_content, expected_output)
