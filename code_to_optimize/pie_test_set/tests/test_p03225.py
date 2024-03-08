from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03225_0():
    input_content = "5 4\n#.##\n.##.\n#...\n..##\n...#"
    expected_output = "3"
    run_pie_test_case("../p03225.py", input_content, expected_output)


def test_problem_p03225_1():
    input_content = "5 4\n#.##\n.##.\n#...\n..##\n...#"
    expected_output = "3"
    run_pie_test_case("../p03225.py", input_content, expected_output)


def test_problem_p03225_2():
    input_content = "13 27\n......#.........#.......#..\n...#.....###..\n..............#####...##...\n...#######......#...#######\n...#.....#.....###...#...#.\n...#######....#.#.#.#.###.#\n..............#.#.#...#.#..\n.#.#.#...###..\n...........#...#...#######\n..#######..#...#...#.....#\n..#.....#..#...#...#.###.#\n..#######..#...#...#.#.#.#\n..........##...#...#.#####"
    expected_output = "870"
    run_pie_test_case("../p03225.py", input_content, expected_output)


def test_problem_p03225_3():
    input_content = "5 4\n.##\n.##.\n...\n..##\n...#"
    expected_output = "3"
    run_pie_test_case("../p03225.py", input_content, expected_output)
