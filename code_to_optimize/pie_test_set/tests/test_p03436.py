from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03436_0():
    input_content = "3 3\n..#\n#..\n..."
    expected_output = "2"
    run_pie_test_case("../p03436.py", input_content, expected_output)


def test_problem_p03436_1():
    input_content = "10 37\n.....................................\n...#...####...####..###...###...###..\n..#.#..#...#.##....#...#.#...#.#...#.\n..#.#..#...#.#.....#...#.#...#.#...#.\n.#...#.#..##.#.....#...#.#.###.#.###.\n.#####.####..#.....#...#..##....##...\n.#...#.#...#.#.....#...#.#...#.#...#.\n.#...#.#...#.##....#...#.#...#.#...#.\n.#...#.####...####..###...###...###..\n....................................."
    expected_output = "209"
    run_pie_test_case("../p03436.py", input_content, expected_output)


def test_problem_p03436_2():
    input_content = "3 3\n..#\n#..\n..."
    expected_output = "2"
    run_pie_test_case("../p03436.py", input_content, expected_output)


def test_problem_p03436_3():
    input_content = "3 3\n..#\n..\n..."
    expected_output = "2"
    run_pie_test_case("../p03436.py", input_content, expected_output)
