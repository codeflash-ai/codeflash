from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03607_0():
    input_content = "3\n6\n2\n6"
    expected_output = "1"
    run_pie_test_case("../p03607.py", input_content, expected_output)


def test_problem_p03607_1():
    input_content = "3\n6\n2\n6"
    expected_output = "1"
    run_pie_test_case("../p03607.py", input_content, expected_output)


def test_problem_p03607_2():
    input_content = "4\n2\n5\n5\n2"
    expected_output = "0"
    run_pie_test_case("../p03607.py", input_content, expected_output)


def test_problem_p03607_3():
    input_content = "6\n12\n22\n16\n22\n18\n12"
    expected_output = "2"
    run_pie_test_case("../p03607.py", input_content, expected_output)
