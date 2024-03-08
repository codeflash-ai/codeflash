from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00210_0():
    input_content = "10 3\n##########\n#E.......X\n##########\n4 4\n####\n#N.#\n#..X\n####\n5 5\n#####\n#N..#\n###.X\n#S..#\n#####\n6 6\n######\n#..#X#\n#.EE.#\n####N#\n#....#\n######\n8 8\n##X#####\n#....E.#\n#####.##\n#.#...##\n#.W.#..#\n#.#.N#.X\n#X##.#.#\n########\n0 0"
    expected_output = "8\nNA\n9\n16\n10"
    run_pie_test_case("../p00210.py", input_content, expected_output)


def test_problem_p00210_1():
    input_content = "10 3\n##########\n#E.......X\n##########\n4 4\n####\n#N.#\n#..X\n####\n5 5\n#####\n#N..#\n###.X\n#S..#\n#####\n6 6\n######\n#..#X#\n#.EE.#\n####N#\n#....#\n######\n8 8\n##X#####\n#....E.#\n#####.##\n#.#...##\n#.W.#..#\n#.#.N#.X\n#X##.#.#\n########\n0 0"
    expected_output = "8\nNA\n9\n16\n10"
    run_pie_test_case("../p00210.py", input_content, expected_output)


def test_problem_p00210_2():
    input_content = "10 3\n\nE.......X\n\n4 4\n\nN.#\n..X\n\n5 5\n\nN..#\n.X\nS..#\n\n6 6\n\n..#X#\n.EE.#\nN#\n....#\n\n8 8\nX#####\n....E.#\n.##\n.#...##\n.W.#..#\n.#.N#.X\nX##.#.#\n\n0 0"
    expected_output = "8\nNA\n9\n16\n10"
    run_pie_test_case("../p00210.py", input_content, expected_output)
