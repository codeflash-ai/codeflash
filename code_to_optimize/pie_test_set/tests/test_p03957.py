from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03957_0():
    input_content = "CODEFESTIVAL"
    expected_output = "Yes"
    run_pie_test_case("../p03957.py", input_content, expected_output)


def test_problem_p03957_1():
    input_content = "CF"
    expected_output = "Yes"
    run_pie_test_case("../p03957.py", input_content, expected_output)


def test_problem_p03957_2():
    input_content = "CODEFESTIVAL"
    expected_output = "Yes"
    run_pie_test_case("../p03957.py", input_content, expected_output)


def test_problem_p03957_3():
    input_content = "FCF"
    expected_output = "Yes"
    run_pie_test_case("../p03957.py", input_content, expected_output)


def test_problem_p03957_4():
    input_content = "FESTIVALCODE"
    expected_output = "No"
    run_pie_test_case("../p03957.py", input_content, expected_output)
