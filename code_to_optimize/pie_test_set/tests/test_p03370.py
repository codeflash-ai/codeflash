from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03370_0():
    input_content = "3 1000\n120\n100\n140"
    expected_output = "9"
    run_pie_test_case("../p03370.py", input_content, expected_output)


def test_problem_p03370_1():
    input_content = "3 1000\n120\n100\n140"
    expected_output = "9"
    run_pie_test_case("../p03370.py", input_content, expected_output)


def test_problem_p03370_2():
    input_content = "5 3000\n150\n130\n150\n130\n110"
    expected_output = "26"
    run_pie_test_case("../p03370.py", input_content, expected_output)


def test_problem_p03370_3():
    input_content = "4 360\n90\n90\n90\n90"
    expected_output = "4"
    run_pie_test_case("../p03370.py", input_content, expected_output)
