from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03672_0():
    input_content = "abaababaab"
    expected_output = "6"
    run_pie_test_case("../p03672.py", input_content, expected_output)


def test_problem_p03672_1():
    input_content = "abaababaab"
    expected_output = "6"
    run_pie_test_case("../p03672.py", input_content, expected_output)


def test_problem_p03672_2():
    input_content = "abcabcabcabc"
    expected_output = "6"
    run_pie_test_case("../p03672.py", input_content, expected_output)


def test_problem_p03672_3():
    input_content = "xxxx"
    expected_output = "2"
    run_pie_test_case("../p03672.py", input_content, expected_output)


def test_problem_p03672_4():
    input_content = "akasakaakasakasakaakas"
    expected_output = "14"
    run_pie_test_case("../p03672.py", input_content, expected_output)
