from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03577_0():
    input_content = "CODEFESTIVAL"
    expected_output = "CODE"
    run_pie_test_case("../p03577.py", input_content, expected_output)


def test_problem_p03577_1():
    input_content = "YAKINIKUFESTIVAL"
    expected_output = "YAKINIKU"
    run_pie_test_case("../p03577.py", input_content, expected_output)


def test_problem_p03577_2():
    input_content = "CODEFESTIVALFESTIVAL"
    expected_output = "CODEFESTIVAL"
    run_pie_test_case("../p03577.py", input_content, expected_output)


def test_problem_p03577_3():
    input_content = "CODEFESTIVAL"
    expected_output = "CODE"
    run_pie_test_case("../p03577.py", input_content, expected_output)
