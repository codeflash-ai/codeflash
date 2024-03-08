from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03433_0():
    input_content = "2018\n218"
    expected_output = "Yes"
    run_pie_test_case("../p03433.py", input_content, expected_output)


def test_problem_p03433_1():
    input_content = "2763\n0"
    expected_output = "No"
    run_pie_test_case("../p03433.py", input_content, expected_output)


def test_problem_p03433_2():
    input_content = "2018\n218"
    expected_output = "Yes"
    run_pie_test_case("../p03433.py", input_content, expected_output)


def test_problem_p03433_3():
    input_content = "37\n514"
    expected_output = "Yes"
    run_pie_test_case("../p03433.py", input_content, expected_output)
