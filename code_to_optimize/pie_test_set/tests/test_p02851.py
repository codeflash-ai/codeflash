from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02851_0():
    input_content = "5 4\n1 4 2 3 5"
    expected_output = "4"
    run_pie_test_case("../p02851.py", input_content, expected_output)


def test_problem_p02851_1():
    input_content = "5 4\n1 4 2 3 5"
    expected_output = "4"
    run_pie_test_case("../p02851.py", input_content, expected_output)


def test_problem_p02851_2():
    input_content = "10 7\n14 15 92 65 35 89 79 32 38 46"
    expected_output = "8"
    run_pie_test_case("../p02851.py", input_content, expected_output)


def test_problem_p02851_3():
    input_content = "8 4\n4 2 4 2 4 2 4 2"
    expected_output = "7"
    run_pie_test_case("../p02851.py", input_content, expected_output)
