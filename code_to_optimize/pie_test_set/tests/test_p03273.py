from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03273_0():
    input_content = "4 4\n##.#\n....\n##.#\n.#.#"
    expected_output = "###\n###\n.##"
    run_pie_test_case("../p03273.py", input_content, expected_output)


def test_problem_p03273_1():
    input_content = "4 4\n##.#\n....\n##.#\n.#.#"
    expected_output = "###\n###\n.##"
    run_pie_test_case("../p03273.py", input_content, expected_output)


def test_problem_p03273_2():
    input_content = "4 5\n.....\n.....\n..#..\n....."
    expected_output = ""
    run_pie_test_case("../p03273.py", input_content, expected_output)


def test_problem_p03273_3():
    input_content = "7 6\n......\n....#.\n.#....\n..#...\n..#...\n......\n.#..#."
    expected_output = "..#\n..\n.#.\n.#.\n.#"
    run_pie_test_case("../p03273.py", input_content, expected_output)


def test_problem_p03273_4():
    input_content = "4 4\n.#\n....\n.#\n.#.#"
    expected_output = ".##"
    run_pie_test_case("../p03273.py", input_content, expected_output)


def test_problem_p03273_5():
    input_content = "3 3\n..\n.#.\n..#"
    expected_output = "..\n.#.\n..#"
    run_pie_test_case("../p03273.py", input_content, expected_output)
