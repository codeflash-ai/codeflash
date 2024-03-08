from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02644_0():
    input_content = "3 5 2\n3 2 3 4\n.....\n.@..@\n..@.."
    expected_output = "5"
    run_pie_test_case("../p02644.py", input_content, expected_output)


def test_problem_p02644_1():
    input_content = "3 5 2\n3 2 3 4\n.....\n.@..@\n..@.."
    expected_output = "5"
    run_pie_test_case("../p02644.py", input_content, expected_output)


def test_problem_p02644_2():
    input_content = "3 3 1\n2 1 2 3\n.@.\n.@.\n.@."
    expected_output = "-1"
    run_pie_test_case("../p02644.py", input_content, expected_output)


def test_problem_p02644_3():
    input_content = "1 6 4\n1 1 1 6\n......"
    expected_output = "2"
    run_pie_test_case("../p02644.py", input_content, expected_output)
