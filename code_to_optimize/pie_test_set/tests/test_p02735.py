from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02735_0():
    input_content = "3 3\n.##\n.#.\n##."
    expected_output = "1"
    run_pie_test_case("../p02735.py", input_content, expected_output)


def test_problem_p02735_1():
    input_content = "2 2\n.\n.#"
    expected_output = "2"
    run_pie_test_case("../p02735.py", input_content, expected_output)


def test_problem_p02735_2():
    input_content = "4 4\n..##\n...\n.\n."
    expected_output = "0"
    run_pie_test_case("../p02735.py", input_content, expected_output)


def test_problem_p02735_3():
    input_content = "5 5\n.#.#.\n.#.#\n.#.#.\n.#.#\n.#.#."
    expected_output = "4"
    run_pie_test_case("../p02735.py", input_content, expected_output)


def test_problem_p02735_4():
    input_content = "3 3\n.##\n.#.\n##."
    expected_output = "1"
    run_pie_test_case("../p02735.py", input_content, expected_output)


def test_problem_p02735_5():
    input_content = "3 3\n.##\n.#.\n."
    expected_output = "1"
    run_pie_test_case("../p02735.py", input_content, expected_output)
