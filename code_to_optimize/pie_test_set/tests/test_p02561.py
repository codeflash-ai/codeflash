from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02561_0():
    input_content = "3 3\n#..\n..#\n..."
    expected_output = "3\n#><\nvv#\n^^."
    run_pie_test_case("../p02561.py", input_content, expected_output)


def test_problem_p02561_1():
    input_content = "3 3\n#..\n..#\n..."
    expected_output = "3\n#><\nvv#\n^^."
    run_pie_test_case("../p02561.py", input_content, expected_output)


def test_problem_p02561_2():
    input_content = "3 3\n..\n..#\n..."
    expected_output = "3\n><\nvv#\n^^."
    run_pie_test_case("../p02561.py", input_content, expected_output)
