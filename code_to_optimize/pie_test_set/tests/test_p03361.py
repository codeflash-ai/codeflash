from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03361_0():
    input_content = "3 3\n.#.\n###\n.#."
    expected_output = "Yes"
    run_pie_test_case("../p03361.py", input_content, expected_output)


def test_problem_p03361_1():
    input_content = "3 3\n.#.\n###\n.#."
    expected_output = "Yes"
    run_pie_test_case("../p03361.py", input_content, expected_output)


def test_problem_p03361_2():
    input_content = "5 5\n.#.#\n.#.#.\n.#.#\n.#.#.\n.#.#"
    expected_output = "No"
    run_pie_test_case("../p03361.py", input_content, expected_output)


def test_problem_p03361_3():
    input_content = "11 11\n...#####...\n.##.....##.\n..##.##..#\n..##.##..#\n.........#\n...###...#\n.#########.\n.#.#.#.#.#.\n.#.#.#.##\n..##.#.##..\n.##..#..##."
    expected_output = "Yes"
    run_pie_test_case("../p03361.py", input_content, expected_output)


def test_problem_p03361_4():
    input_content = "3 3\n.#.\n\n.#."
    expected_output = "Yes"
    run_pie_test_case("../p03361.py", input_content, expected_output)
