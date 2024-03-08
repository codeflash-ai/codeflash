from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02910_0():
    input_content = "RUDLUDR"
    expected_output = "Yes"
    run_pie_test_case("../p02910.py", input_content, expected_output)


def test_problem_p02910_1():
    input_content = "RUDLUDR"
    expected_output = "Yes"
    run_pie_test_case("../p02910.py", input_content, expected_output)


def test_problem_p02910_2():
    input_content = "DULL"
    expected_output = "No"
    run_pie_test_case("../p02910.py", input_content, expected_output)


def test_problem_p02910_3():
    input_content = "RDULULDURURLRDULRLR"
    expected_output = "Yes"
    run_pie_test_case("../p02910.py", input_content, expected_output)


def test_problem_p02910_4():
    input_content = "UUUUUUUUUUUUUUU"
    expected_output = "Yes"
    run_pie_test_case("../p02910.py", input_content, expected_output)


def test_problem_p02910_5():
    input_content = "ULURU"
    expected_output = "No"
    run_pie_test_case("../p02910.py", input_content, expected_output)
