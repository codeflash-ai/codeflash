from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03486_0():
    input_content = "yx\naxy"
    expected_output = "Yes"
    run_pie_test_case("../p03486.py", input_content, expected_output)


def test_problem_p03486_1():
    input_content = "ratcode\natlas"
    expected_output = "Yes"
    run_pie_test_case("../p03486.py", input_content, expected_output)


def test_problem_p03486_2():
    input_content = "zzz\nzzz"
    expected_output = "No"
    run_pie_test_case("../p03486.py", input_content, expected_output)


def test_problem_p03486_3():
    input_content = "w\nww"
    expected_output = "Yes"
    run_pie_test_case("../p03486.py", input_content, expected_output)


def test_problem_p03486_4():
    input_content = "cd\nabc"
    expected_output = "No"
    run_pie_test_case("../p03486.py", input_content, expected_output)


def test_problem_p03486_5():
    input_content = "yx\naxy"
    expected_output = "Yes"
    run_pie_test_case("../p03486.py", input_content, expected_output)
