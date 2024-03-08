from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03574_0():
    input_content = "3 5\n.....\n.#.#.\n....."
    expected_output = "11211\n1#2#1\n11211"
    run_pie_test_case("../p03574.py", input_content, expected_output)


def test_problem_p03574_1():
    input_content = "3 5\n.....\n.#.#.\n....."
    expected_output = "11211\n1#2#1\n11211"
    run_pie_test_case("../p03574.py", input_content, expected_output)


def test_problem_p03574_2():
    input_content = "3 5"
    expected_output = ""
    run_pie_test_case("../p03574.py", input_content, expected_output)


def test_problem_p03574_3():
    input_content = "6 6\n.\n.#.##\n.#\n.#..#.\n.##..\n.#..."
    expected_output = "3\n8#7##\n5#\n4#65#2\n5##21\n4#310"
    run_pie_test_case("../p03574.py", input_content, expected_output)
