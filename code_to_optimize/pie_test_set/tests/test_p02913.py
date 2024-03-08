from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02913_0():
    input_content = "5\nababa"
    expected_output = "2"
    run_pie_test_case("../p02913.py", input_content, expected_output)


def test_problem_p02913_1():
    input_content = "13\nstrangeorange"
    expected_output = "5"
    run_pie_test_case("../p02913.py", input_content, expected_output)


def test_problem_p02913_2():
    input_content = "2\nxy"
    expected_output = "0"
    run_pie_test_case("../p02913.py", input_content, expected_output)


def test_problem_p02913_3():
    input_content = "5\nababa"
    expected_output = "2"
    run_pie_test_case("../p02913.py", input_content, expected_output)
