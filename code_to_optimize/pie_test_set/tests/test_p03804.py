from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03804_0():
    input_content = "3 2\n#.#\n.#.\n#.#\n#.\n.#"
    expected_output = "Yes"
    run_pie_test_case("../p03804.py", input_content, expected_output)


def test_problem_p03804_1():
    input_content = "3 2\n#.#\n.#.\n#.#\n#.\n.#"
    expected_output = "Yes"
    run_pie_test_case("../p03804.py", input_content, expected_output)


def test_problem_p03804_2():
    input_content = "4 1\n....\n....\n....\n...."
    expected_output = "No"
    run_pie_test_case("../p03804.py", input_content, expected_output)


def test_problem_p03804_3():
    input_content = "3 2\n.#\n.#.\n.#\n.\n.#"
    expected_output = "Yes"
    run_pie_test_case("../p03804.py", input_content, expected_output)
