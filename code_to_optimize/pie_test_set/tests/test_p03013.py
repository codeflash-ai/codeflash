from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03013_0():
    input_content = "6 1\n3"
    expected_output = "4"
    run_pie_test_case("../p03013.py", input_content, expected_output)


def test_problem_p03013_1():
    input_content = "100 5\n1\n23\n45\n67\n89"
    expected_output = "608200469"
    run_pie_test_case("../p03013.py", input_content, expected_output)


def test_problem_p03013_2():
    input_content = "10 2\n4\n5"
    expected_output = "0"
    run_pie_test_case("../p03013.py", input_content, expected_output)


def test_problem_p03013_3():
    input_content = "6 1\n3"
    expected_output = "4"
    run_pie_test_case("../p03013.py", input_content, expected_output)
