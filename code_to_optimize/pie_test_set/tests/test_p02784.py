from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02784_0():
    input_content = "10 3\n4 5 6"
    expected_output = "Yes"
    run_pie_test_case("../p02784.py", input_content, expected_output)


def test_problem_p02784_1():
    input_content = "211 5\n31 41 59 26 53"
    expected_output = "No"
    run_pie_test_case("../p02784.py", input_content, expected_output)


def test_problem_p02784_2():
    input_content = "10 3\n4 5 6"
    expected_output = "Yes"
    run_pie_test_case("../p02784.py", input_content, expected_output)


def test_problem_p02784_3():
    input_content = "20 3\n4 5 6"
    expected_output = "No"
    run_pie_test_case("../p02784.py", input_content, expected_output)


def test_problem_p02784_4():
    input_content = "210 5\n31 41 59 26 53"
    expected_output = "Yes"
    run_pie_test_case("../p02784.py", input_content, expected_output)
