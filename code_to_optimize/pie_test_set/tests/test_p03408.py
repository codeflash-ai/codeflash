from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03408_0():
    input_content = "3\napple\norange\napple\n1\ngrape"
    expected_output = "2"
    run_pie_test_case("../p03408.py", input_content, expected_output)


def test_problem_p03408_1():
    input_content = "3\napple\norange\napple\n5\napple\napple\napple\napple\napple"
    expected_output = "1"
    run_pie_test_case("../p03408.py", input_content, expected_output)


def test_problem_p03408_2():
    input_content = "3\napple\norange\napple\n1\ngrape"
    expected_output = "2"
    run_pie_test_case("../p03408.py", input_content, expected_output)


def test_problem_p03408_3():
    input_content = "6\nred\nred\nblue\nyellow\nyellow\nred\n5\nred\nred\nyellow\ngreen\nblue"
    expected_output = "1"
    run_pie_test_case("../p03408.py", input_content, expected_output)


def test_problem_p03408_4():
    input_content = "1\nvoldemort\n10\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort\nvoldemort"
    expected_output = "0"
    run_pie_test_case("../p03408.py", input_content, expected_output)
