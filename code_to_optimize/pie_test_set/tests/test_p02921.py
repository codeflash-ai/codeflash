from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02921_0():
    input_content = "CSS\nCSR"
    expected_output = "2"
    run_pie_test_case("../p02921.py", input_content, expected_output)


def test_problem_p02921_1():
    input_content = "CSS\nCSR"
    expected_output = "2"
    run_pie_test_case("../p02921.py", input_content, expected_output)


def test_problem_p02921_2():
    input_content = "RRR\nSSS"
    expected_output = "0"
    run_pie_test_case("../p02921.py", input_content, expected_output)


def test_problem_p02921_3():
    input_content = "SSR\nSSR"
    expected_output = "3"
    run_pie_test_case("../p02921.py", input_content, expected_output)
