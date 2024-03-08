from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03951_0():
    input_content = "3\nabc\ncde"
    expected_output = "5"
    run_pie_test_case("../p03951.py", input_content, expected_output)


def test_problem_p03951_1():
    input_content = "3\nabc\ncde"
    expected_output = "5"
    run_pie_test_case("../p03951.py", input_content, expected_output)


def test_problem_p03951_2():
    input_content = "1\na\nz"
    expected_output = "2"
    run_pie_test_case("../p03951.py", input_content, expected_output)


def test_problem_p03951_3():
    input_content = "4\nexpr\nexpr"
    expected_output = "4"
    run_pie_test_case("../p03951.py", input_content, expected_output)
