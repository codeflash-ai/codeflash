from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02584_0():
    input_content = "6 2 4"
    expected_output = "2"
    run_pie_test_case("../p02584.py", input_content, expected_output)


def test_problem_p02584_1():
    input_content = "1000000000000000 1000000000000000 1000000000000000"
    expected_output = "1000000000000000"
    run_pie_test_case("../p02584.py", input_content, expected_output)


def test_problem_p02584_2():
    input_content = "6 2 4"
    expected_output = "2"
    run_pie_test_case("../p02584.py", input_content, expected_output)


def test_problem_p02584_3():
    input_content = "10 1 2"
    expected_output = "8"
    run_pie_test_case("../p02584.py", input_content, expected_output)


def test_problem_p02584_4():
    input_content = "7 4 3"
    expected_output = "1"
    run_pie_test_case("../p02584.py", input_content, expected_output)
