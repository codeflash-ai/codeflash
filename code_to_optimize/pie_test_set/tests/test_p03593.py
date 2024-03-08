from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03593_0():
    input_content = "3 4\naabb\naabb\naacc"
    expected_output = "Yes"
    run_pie_test_case("../p03593.py", input_content, expected_output)


def test_problem_p03593_1():
    input_content = "2 5\nabxba\nabyba"
    expected_output = "No"
    run_pie_test_case("../p03593.py", input_content, expected_output)


def test_problem_p03593_2():
    input_content = "5 1\nt\nw\ne\ne\nt"
    expected_output = "Yes"
    run_pie_test_case("../p03593.py", input_content, expected_output)


def test_problem_p03593_3():
    input_content = "2 2\naa\nbb"
    expected_output = "No"
    run_pie_test_case("../p03593.py", input_content, expected_output)


def test_problem_p03593_4():
    input_content = "1 1\nz"
    expected_output = "Yes"
    run_pie_test_case("../p03593.py", input_content, expected_output)


def test_problem_p03593_5():
    input_content = "3 4\naabb\naabb\naacc"
    expected_output = "Yes"
    run_pie_test_case("../p03593.py", input_content, expected_output)
