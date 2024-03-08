from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02721_0():
    input_content = "11 3 2\nooxxxoxxxoo"
    expected_output = "6"
    run_pie_test_case("../p02721.py", input_content, expected_output)


def test_problem_p02721_1():
    input_content = "5 2 3\nooxoo"
    expected_output = "1\n5"
    run_pie_test_case("../p02721.py", input_content, expected_output)


def test_problem_p02721_2():
    input_content = "11 3 2\nooxxxoxxxoo"
    expected_output = "6"
    run_pie_test_case("../p02721.py", input_content, expected_output)


def test_problem_p02721_3():
    input_content = "16 4 3\nooxxoxoxxxoxoxxo"
    expected_output = "11\n16"
    run_pie_test_case("../p02721.py", input_content, expected_output)
