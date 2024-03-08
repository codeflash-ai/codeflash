from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03239_0():
    input_content = "3 70\n7 60\n1 80\n4 50"
    expected_output = "4"
    run_pie_test_case("../p03239.py", input_content, expected_output)


def test_problem_p03239_1():
    input_content = "4 3\n1 1000\n2 4\n3 1000\n4 500"
    expected_output = "TLE"
    run_pie_test_case("../p03239.py", input_content, expected_output)


def test_problem_p03239_2():
    input_content = "5 9\n25 8\n5 9\n4 10\n1000 1000\n6 1"
    expected_output = "5"
    run_pie_test_case("../p03239.py", input_content, expected_output)


def test_problem_p03239_3():
    input_content = "3 70\n7 60\n1 80\n4 50"
    expected_output = "4"
    run_pie_test_case("../p03239.py", input_content, expected_output)
