from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03712_0():
    input_content = "2 3\nabc\narc"
    expected_output = "#####\n#abc#\n#arc#\n#####"
    run_pie_test_case("../p03712.py", input_content, expected_output)


def test_problem_p03712_1():
    input_content = "2 3\nabc\narc"
    expected_output = "#####\n#abc#\n#arc#\n#####"
    run_pie_test_case("../p03712.py", input_content, expected_output)


def test_problem_p03712_2():
    input_content = "1 1\nz"
    expected_output = "z#"
    run_pie_test_case("../p03712.py", input_content, expected_output)
