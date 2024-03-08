from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02579_0():
    input_content = "4 4\n1 1\n4 4\n..#.\n..#.\n.#..\n.#.."
    expected_output = "1"
    run_pie_test_case("../p02579.py", input_content, expected_output)


def test_problem_p02579_1():
    input_content = "4 4\n1 1\n4 4\n..#.\n..#.\n.#..\n.#.."
    expected_output = "1"
    run_pie_test_case("../p02579.py", input_content, expected_output)


def test_problem_p02579_2():
    input_content = "4 4\n1 4\n4 1\n.##.\n\n\n.##."
    expected_output = "-1"
    run_pie_test_case("../p02579.py", input_content, expected_output)


def test_problem_p02579_3():
    input_content = "4 5\n1 2\n2 5\n.###\n.\n..##\n..##"
    expected_output = "2"
    run_pie_test_case("../p02579.py", input_content, expected_output)


def test_problem_p02579_4():
    input_content = "4 4\n2 2\n3 3\n....\n....\n....\n...."
    expected_output = "0"
    run_pie_test_case("../p02579.py", input_content, expected_output)
