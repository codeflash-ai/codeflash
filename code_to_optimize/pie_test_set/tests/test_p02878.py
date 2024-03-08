from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02878_0():
    input_content = "5 1 3"
    expected_output = "4"
    run_pie_test_case("../p02878.py", input_content, expected_output)


def test_problem_p02878_1():
    input_content = "5 1 3"
    expected_output = "4"
    run_pie_test_case("../p02878.py", input_content, expected_output)


def test_problem_p02878_2():
    input_content = "1000000 100000 200000"
    expected_output = "758840509"
    run_pie_test_case("../p02878.py", input_content, expected_output)


def test_problem_p02878_3():
    input_content = "10 4 6"
    expected_output = "197"
    run_pie_test_case("../p02878.py", input_content, expected_output)


def test_problem_p02878_4():
    input_content = "10 0 0"
    expected_output = "1"
    run_pie_test_case("../p02878.py", input_content, expected_output)
