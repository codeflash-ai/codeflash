from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03744_0():
    input_content = "3 10\n10 10\n20 5\n4 3"
    expected_output = "10.0000000\n15.0000000\n13.2000000"
    run_pie_test_case("../p03744.py", input_content, expected_output)


def test_problem_p03744_1():
    input_content = "4 15\n1000000000 15\n9 5\n8 6\n7 4"
    expected_output = "1000000000.0000000\n666666669.6666666\n400000005.0000000\n293333338.8666667"
    run_pie_test_case("../p03744.py", input_content, expected_output)


def test_problem_p03744_2():
    input_content = "4 15\n0 15\n2 5\n3 6\n4 4"
    expected_output = "0.0000000\n0.6666667\n1.8666667\n2.9333333"
    run_pie_test_case("../p03744.py", input_content, expected_output)


def test_problem_p03744_3():
    input_content = "3 10\n10 10\n20 5\n4 3"
    expected_output = "10.0000000\n15.0000000\n13.2000000"
    run_pie_test_case("../p03744.py", input_content, expected_output)
