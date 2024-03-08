from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03470_0():
    input_content = "4\n10\n8\n8\n6"
    expected_output = "3"
    run_pie_test_case("../p03470.py", input_content, expected_output)


def test_problem_p03470_1():
    input_content = "7\n50\n30\n50\n100\n50\n80\n30"
    expected_output = "4"
    run_pie_test_case("../p03470.py", input_content, expected_output)


def test_problem_p03470_2():
    input_content = "3\n15\n15\n15"
    expected_output = "1"
    run_pie_test_case("../p03470.py", input_content, expected_output)


def test_problem_p03470_3():
    input_content = "4\n10\n8\n8\n6"
    expected_output = "3"
    run_pie_test_case("../p03470.py", input_content, expected_output)
