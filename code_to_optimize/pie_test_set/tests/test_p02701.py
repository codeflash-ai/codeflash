from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02701_0():
    input_content = "3\napple\norange\napple"
    expected_output = "2"
    run_pie_test_case("../p02701.py", input_content, expected_output)


def test_problem_p02701_1():
    input_content = "4\naaaa\na\naaa\naa"
    expected_output = "4"
    run_pie_test_case("../p02701.py", input_content, expected_output)


def test_problem_p02701_2():
    input_content = "3\napple\norange\napple"
    expected_output = "2"
    run_pie_test_case("../p02701.py", input_content, expected_output)


def test_problem_p02701_3():
    input_content = "5\ngrape\ngrape\ngrape\ngrape\ngrape"
    expected_output = "1"
    run_pie_test_case("../p02701.py", input_content, expected_output)
