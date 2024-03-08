from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02819_0():
    input_content = "20"
    expected_output = "23"
    run_pie_test_case("../p02819.py", input_content, expected_output)


def test_problem_p02819_1():
    input_content = "99992"
    expected_output = "100003"
    run_pie_test_case("../p02819.py", input_content, expected_output)


def test_problem_p02819_2():
    input_content = "20"
    expected_output = "23"
    run_pie_test_case("../p02819.py", input_content, expected_output)


def test_problem_p02819_3():
    input_content = "2"
    expected_output = "2"
    run_pie_test_case("../p02819.py", input_content, expected_output)
