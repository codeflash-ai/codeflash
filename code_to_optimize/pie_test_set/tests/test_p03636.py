from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03636_0():
    input_content = "internationalization"
    expected_output = "i18n"
    run_pie_test_case("../p03636.py", input_content, expected_output)


def test_problem_p03636_1():
    input_content = "smiles"
    expected_output = "s4s"
    run_pie_test_case("../p03636.py", input_content, expected_output)


def test_problem_p03636_2():
    input_content = "internationalization"
    expected_output = "i18n"
    run_pie_test_case("../p03636.py", input_content, expected_output)


def test_problem_p03636_3():
    input_content = "xyz"
    expected_output = "x1z"
    run_pie_test_case("../p03636.py", input_content, expected_output)
