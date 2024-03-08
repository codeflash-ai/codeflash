from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03014_0():
    input_content = "4 6\n#..#..\n.....#\n....#.\n#.#..."
    expected_output = "8"
    run_pie_test_case("../p03014.py", input_content, expected_output)


def test_problem_p03014_1():
    input_content = "4 6\n#..#..\n.....#\n....#.\n#.#..."
    expected_output = "8"
    run_pie_test_case("../p03014.py", input_content, expected_output)


def test_problem_p03014_2():
    input_content = "4 6\n..#..\n.....#\n....#.\n.#..."
    expected_output = "8"
    run_pie_test_case("../p03014.py", input_content, expected_output)


def test_problem_p03014_3():
    input_content = "8 8\n..#...#.\n....#...\n......\n..###..#\n...#..#.\n....#.\n...#...\n.#..#"
    expected_output = "13"
    run_pie_test_case("../p03014.py", input_content, expected_output)
