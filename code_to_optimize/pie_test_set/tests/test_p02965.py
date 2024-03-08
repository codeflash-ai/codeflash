from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02965_0():
    input_content = "2 2"
    expected_output = "3"
    run_pie_test_case("../p02965.py", input_content, expected_output)


def test_problem_p02965_1():
    input_content = "3 2"
    expected_output = "19"
    run_pie_test_case("../p02965.py", input_content, expected_output)


def test_problem_p02965_2():
    input_content = "100000 50000"
    expected_output = "3463133"
    run_pie_test_case("../p02965.py", input_content, expected_output)


def test_problem_p02965_3():
    input_content = "10 10"
    expected_output = "211428932"
    run_pie_test_case("../p02965.py", input_content, expected_output)


def test_problem_p02965_4():
    input_content = "2 2"
    expected_output = "3"
    run_pie_test_case("../p02965.py", input_content, expected_output)
