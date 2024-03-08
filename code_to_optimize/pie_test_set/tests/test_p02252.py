from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02252_0():
    input_content = "3 50\n60 10\n100 20\n120 30"
    expected_output = "240"
    run_pie_test_case("../p02252.py", input_content, expected_output)


def test_problem_p02252_1():
    input_content = "1 100\n100000 100000"
    expected_output = "100"
    run_pie_test_case("../p02252.py", input_content, expected_output)


def test_problem_p02252_2():
    input_content = "3 50\n60 10\n100 20\n120 30"
    expected_output = "240"
    run_pie_test_case("../p02252.py", input_content, expected_output)


def test_problem_p02252_3():
    input_content = "3 50\n60 13\n100 23\n120 33"
    expected_output = "210.90909091"
    run_pie_test_case("../p02252.py", input_content, expected_output)
