from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03476_0():
    input_content = "1\n3 7"
    expected_output = "2"
    run_pie_test_case("../p03476.py", input_content, expected_output)


def test_problem_p03476_1():
    input_content = "1\n3 7"
    expected_output = "2"
    run_pie_test_case("../p03476.py", input_content, expected_output)


def test_problem_p03476_2():
    input_content = "4\n13 13\n7 11\n7 11\n2017 2017"
    expected_output = "1\n0\n0\n1"
    run_pie_test_case("../p03476.py", input_content, expected_output)


def test_problem_p03476_3():
    input_content = "6\n1 53\n13 91\n37 55\n19 51\n73 91\n13 49"
    expected_output = "4\n4\n1\n1\n1\n2"
    run_pie_test_case("../p03476.py", input_content, expected_output)
