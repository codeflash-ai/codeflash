from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01631_0():
    input_content = "6\nAIZU 10\nLINER 6\nLINE 4\nALL 2\nAS 1\nCIEL 10\nASLA\nCILI\nIRZN\nELEU\n21"
    expected_output = "40"
    run_pie_test_case("../p01631.py", input_content, expected_output)


def test_problem_p01631_1():
    input_content = "6\nAIZU 10\nLINER 6\nLINE 4\nALL 2\nAS 1\nCIEL 10\nASLA\nCILI\nIRZN\nELEU\n21"
    expected_output = "40"
    run_pie_test_case("../p01631.py", input_content, expected_output)
