from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03157_0():
    input_content = "3 3\n.#.\n..#\n#.."
    expected_output = "10"
    run_pie_test_case("../p03157.py", input_content, expected_output)


def test_problem_p03157_1():
    input_content = "3 3\n.#.\n..#\n#.."
    expected_output = "10"
    run_pie_test_case("../p03157.py", input_content, expected_output)


def test_problem_p03157_2():
    input_content = "4 3\n\n\n..."
    expected_output = "6"
    run_pie_test_case("../p03157.py", input_content, expected_output)


def test_problem_p03157_3():
    input_content = "3 3\n.#.\n..#\n.."
    expected_output = "10"
    run_pie_test_case("../p03157.py", input_content, expected_output)


def test_problem_p03157_4():
    input_content = "2 4\n....\n...."
    expected_output = "0"
    run_pie_test_case("../p03157.py", input_content, expected_output)
