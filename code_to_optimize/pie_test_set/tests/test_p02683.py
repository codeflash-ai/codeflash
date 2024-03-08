from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02683_0():
    input_content = "3 3 10\n60 2 2 4\n70 8 7 9\n50 2 3 9"
    expected_output = "120"
    run_pie_test_case("../p02683.py", input_content, expected_output)


def test_problem_p02683_1():
    input_content = "3 3 10\n100 3 1 4\n100 1 5 9\n100 2 6 5"
    expected_output = "-1"
    run_pie_test_case("../p02683.py", input_content, expected_output)


def test_problem_p02683_2():
    input_content = "3 3 10\n60 2 2 4\n70 8 7 9\n50 2 3 9"
    expected_output = "120"
    run_pie_test_case("../p02683.py", input_content, expected_output)


def test_problem_p02683_3():
    input_content = "8 5 22\n100 3 7 5 3 1\n164 4 5 2 7 8\n334 7 2 7 2 9\n234 4 7 2 8 2\n541 5 4 3 3 6\n235 4 8 6 9 7\n394 3 6 1 6 2\n872 8 4 3 7 2"
    expected_output = "1067"
    run_pie_test_case("../p02683.py", input_content, expected_output)
