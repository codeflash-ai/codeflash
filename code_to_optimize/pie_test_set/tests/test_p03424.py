from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03424_0():
    input_content = "6\nG W Y P Y W"
    expected_output = "Four"
    run_pie_test_case("../p03424.py", input_content, expected_output)


def test_problem_p03424_1():
    input_content = "6\nG W Y P Y W"
    expected_output = "Four"
    run_pie_test_case("../p03424.py", input_content, expected_output)


def test_problem_p03424_2():
    input_content = "9\nG W W G P W P G G"
    expected_output = "Three"
    run_pie_test_case("../p03424.py", input_content, expected_output)


def test_problem_p03424_3():
    input_content = "8\nP Y W G Y W Y Y"
    expected_output = "Four"
    run_pie_test_case("../p03424.py", input_content, expected_output)
