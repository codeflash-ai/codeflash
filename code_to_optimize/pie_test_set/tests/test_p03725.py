from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03725_0():
    input_content = "3 3 3\n#.#\n#S.\n###"
    expected_output = "1"
    run_pie_test_case("../p03725.py", input_content, expected_output)


def test_problem_p03725_1():
    input_content = "3 3 3\n#.#\n#S.\n###"
    expected_output = "1"
    run_pie_test_case("../p03725.py", input_content, expected_output)


def test_problem_p03725_2():
    input_content = "3 3 3\n.#\nS."
    expected_output = "1"
    run_pie_test_case("../p03725.py", input_content, expected_output)


def test_problem_p03725_3():
    input_content = "3 3 3\n\nS#"
    expected_output = "2"
    run_pie_test_case("../p03725.py", input_content, expected_output)


def test_problem_p03725_4():
    input_content = "7 7 2\n\n\n...##\nS###\n.#.##\n.###"
    expected_output = "2"
    run_pie_test_case("../p03725.py", input_content, expected_output)
