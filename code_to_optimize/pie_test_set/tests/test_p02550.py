from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02550_0():
    input_content = "6 2 1001"
    expected_output = "1369"
    run_pie_test_case("../p02550.py", input_content, expected_output)


def test_problem_p02550_1():
    input_content = "10000000000 10 99959"
    expected_output = "492443256176507"
    run_pie_test_case("../p02550.py", input_content, expected_output)


def test_problem_p02550_2():
    input_content = "6 2 1001"
    expected_output = "1369"
    run_pie_test_case("../p02550.py", input_content, expected_output)


def test_problem_p02550_3():
    input_content = "1000 2 16"
    expected_output = "6"
    run_pie_test_case("../p02550.py", input_content, expected_output)
