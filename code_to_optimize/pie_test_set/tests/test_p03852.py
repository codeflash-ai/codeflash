from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03852_0():
    input_content = "a"
    expected_output = "vowel"
    run_pie_test_case("../p03852.py", input_content, expected_output)


def test_problem_p03852_1():
    input_content = "s"
    expected_output = "consonant"
    run_pie_test_case("../p03852.py", input_content, expected_output)


def test_problem_p03852_2():
    input_content = "z"
    expected_output = "consonant"
    run_pie_test_case("../p03852.py", input_content, expected_output)


def test_problem_p03852_3():
    input_content = "a"
    expected_output = "vowel"
    run_pie_test_case("../p03852.py", input_content, expected_output)
