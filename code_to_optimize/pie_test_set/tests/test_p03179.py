from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03179_0():
    input_content = "4\n<><"
    expected_output = "5"
    run_pie_test_case("../p03179.py", input_content, expected_output)


def test_problem_p03179_1():
    input_content = "20\n>>>><>>><>><>>><<>>"
    expected_output = "217136290"
    run_pie_test_case("../p03179.py", input_content, expected_output)


def test_problem_p03179_2():
    input_content = "5\n<<<<"
    expected_output = "1"
    run_pie_test_case("../p03179.py", input_content, expected_output)


def test_problem_p03179_3():
    input_content = "4\n<><"
    expected_output = "5"
    run_pie_test_case("../p03179.py", input_content, expected_output)
