from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03431_0():
    input_content = "2 4"
    expected_output = "7"
    run_pie_test_case("../p03431.py", input_content, expected_output)


def test_problem_p03431_1():
    input_content = "8 10"
    expected_output = "46"
    run_pie_test_case("../p03431.py", input_content, expected_output)


def test_problem_p03431_2():
    input_content = "3 7"
    expected_output = "57"
    run_pie_test_case("../p03431.py", input_content, expected_output)


def test_problem_p03431_3():
    input_content = "123456 234567"
    expected_output = "857617983"
    run_pie_test_case("../p03431.py", input_content, expected_output)


def test_problem_p03431_4():
    input_content = "8 3"
    expected_output = "0"
    run_pie_test_case("../p03431.py", input_content, expected_output)


def test_problem_p03431_5():
    input_content = "2 4"
    expected_output = "7"
    run_pie_test_case("../p03431.py", input_content, expected_output)
