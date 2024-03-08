from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02941_0():
    input_content = "3\n1 1 1\n13 5 7"
    expected_output = "4"
    run_pie_test_case("../p02941.py", input_content, expected_output)


def test_problem_p02941_1():
    input_content = "5\n5 6 5 2 1\n9817 1108 6890 4343 8704"
    expected_output = "25"
    run_pie_test_case("../p02941.py", input_content, expected_output)


def test_problem_p02941_2():
    input_content = "4\n1 2 3 4\n2 3 4 5"
    expected_output = "-1"
    run_pie_test_case("../p02941.py", input_content, expected_output)


def test_problem_p02941_3():
    input_content = "3\n1 1 1\n13 5 7"
    expected_output = "4"
    run_pie_test_case("../p02941.py", input_content, expected_output)
