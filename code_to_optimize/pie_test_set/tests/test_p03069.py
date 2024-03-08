from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03069_0():
    input_content = "3\n#.#"
    expected_output = "1"
    run_pie_test_case("../p03069.py", input_content, expected_output)


def test_problem_p03069_1():
    input_content = "3\n.#"
    expected_output = "1"
    run_pie_test_case("../p03069.py", input_content, expected_output)


def test_problem_p03069_2():
    input_content = "5\n.##."
    expected_output = "2"
    run_pie_test_case("../p03069.py", input_content, expected_output)


def test_problem_p03069_3():
    input_content = "9\n........."
    expected_output = "0"
    run_pie_test_case("../p03069.py", input_content, expected_output)


def test_problem_p03069_4():
    input_content = "3\n#.#"
    expected_output = "1"
    run_pie_test_case("../p03069.py", input_content, expected_output)
