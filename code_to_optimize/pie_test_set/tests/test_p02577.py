from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02577_0():
    input_content = "123456789"
    expected_output = "Yes"
    run_pie_test_case("../p02577.py", input_content, expected_output)


def test_problem_p02577_1():
    input_content = "123456789"
    expected_output = "Yes"
    run_pie_test_case("../p02577.py", input_content, expected_output)


def test_problem_p02577_2():
    input_content = "0"
    expected_output = "Yes"
    run_pie_test_case("../p02577.py", input_content, expected_output)


def test_problem_p02577_3():
    input_content = (
        "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280"
    )
    expected_output = "No"
    run_pie_test_case("../p02577.py", input_content, expected_output)
