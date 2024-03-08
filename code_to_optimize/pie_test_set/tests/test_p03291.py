from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03291_0():
    input_content = "A??C"
    expected_output = "8"
    run_pie_test_case("../p03291.py", input_content, expected_output)


def test_problem_p03291_1():
    input_content = "ABCBC"
    expected_output = "3"
    run_pie_test_case("../p03291.py", input_content, expected_output)


def test_problem_p03291_2():
    input_content = "????C?????B??????A???????"
    expected_output = "979596887"
    run_pie_test_case("../p03291.py", input_content, expected_output)


def test_problem_p03291_3():
    input_content = "A??C"
    expected_output = "8"
    run_pie_test_case("../p03291.py", input_content, expected_output)
