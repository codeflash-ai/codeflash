from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03165_0():
    input_content = "axyb\nabyxb"
    expected_output = "axb"
    run_pie_test_case("../p03165.py", input_content, expected_output)


def test_problem_p03165_1():
    input_content = "abracadabra\navadakedavra"
    expected_output = "aaadara"
    run_pie_test_case("../p03165.py", input_content, expected_output)


def test_problem_p03165_2():
    input_content = "a\nz"
    expected_output = ""
    run_pie_test_case("../p03165.py", input_content, expected_output)


def test_problem_p03165_3():
    input_content = "axyb\nabyxb"
    expected_output = "axb"
    run_pie_test_case("../p03165.py", input_content, expected_output)


def test_problem_p03165_4():
    input_content = "aa\nxayaz"
    expected_output = "aa"
    run_pie_test_case("../p03165.py", input_content, expected_output)
