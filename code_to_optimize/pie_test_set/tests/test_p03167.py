from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03167_0():
    input_content = "3 4\n...#\n.#..\n...."
    expected_output = "3"
    run_pie_test_case("../p03167.py", input_content, expected_output)


def test_problem_p03167_1():
    input_content = "5 5\n..#..\n.....\n...#\n.....\n..#.."
    expected_output = "24"
    run_pie_test_case("../p03167.py", input_content, expected_output)


def test_problem_p03167_2():
    input_content = "5 2\n..\n.\n..\n.#\n.."
    expected_output = "0"
    run_pie_test_case("../p03167.py", input_content, expected_output)


def test_problem_p03167_3():
    input_content = "3 4\n...#\n.#..\n...."
    expected_output = "3"
    run_pie_test_case("../p03167.py", input_content, expected_output)


def test_problem_p03167_4():
    input_content = "20 20\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n....................\n...................."
    expected_output = "345263555"
    run_pie_test_case("../p03167.py", input_content, expected_output)
