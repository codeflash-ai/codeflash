from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04040_0():
    input_content = "2 3 1 1"
    expected_output = "2"
    run_pie_test_case("../p04040.py", input_content, expected_output)


def test_problem_p04040_1():
    input_content = "2 3 1 1"
    expected_output = "2"
    run_pie_test_case("../p04040.py", input_content, expected_output)


def test_problem_p04040_2():
    input_content = "100000 100000 99999 99999"
    expected_output = "1"
    run_pie_test_case("../p04040.py", input_content, expected_output)


def test_problem_p04040_3():
    input_content = "100000 100000 44444 55555"
    expected_output = "738162020"
    run_pie_test_case("../p04040.py", input_content, expected_output)


def test_problem_p04040_4():
    input_content = "10 7 3 4"
    expected_output = "3570"
    run_pie_test_case("../p04040.py", input_content, expected_output)
