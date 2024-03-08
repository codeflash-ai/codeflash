from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03053_0():
    input_content = "3 3\n...\n.#.\n..."
    expected_output = "2"
    run_pie_test_case("../p03053.py", input_content, expected_output)


def test_problem_p03053_1():
    input_content = "3 3\n...\n.#.\n..."
    expected_output = "2"
    run_pie_test_case("../p03053.py", input_content, expected_output)


def test_problem_p03053_2():
    input_content = "6 6\n..#..#\n......\n..#..\n......\n.#....\n....#."
    expected_output = "3"
    run_pie_test_case("../p03053.py", input_content, expected_output)
