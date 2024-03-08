from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03738_0():
    input_content = "36\n24"
    expected_output = "GREATER"
    run_pie_test_case("../p03738.py", input_content, expected_output)


def test_problem_p03738_1():
    input_content = "36\n24"
    expected_output = "GREATER"
    run_pie_test_case("../p03738.py", input_content, expected_output)


def test_problem_p03738_2():
    input_content = "9720246\n22516266"
    expected_output = "LESS"
    run_pie_test_case("../p03738.py", input_content, expected_output)


def test_problem_p03738_3():
    input_content = "123456789012345678901234567890\n234567890123456789012345678901"
    expected_output = "LESS"
    run_pie_test_case("../p03738.py", input_content, expected_output)


def test_problem_p03738_4():
    input_content = "850\n3777"
    expected_output = "LESS"
    run_pie_test_case("../p03738.py", input_content, expected_output)
