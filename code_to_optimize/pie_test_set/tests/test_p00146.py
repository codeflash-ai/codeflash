from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00146_0():
    input_content = "2\n1 100 1\n2 200 2"
    expected_output = "1 2"
    run_pie_test_case("../p00146.py", input_content, expected_output)


def test_problem_p00146_1():
    input_content = "5\n13 199 1\n51 1000 1\n37 350 10\n27 300 2\n99 200 1000"
    expected_output = "51 37 27 13 99"
    run_pie_test_case("../p00146.py", input_content, expected_output)


def test_problem_p00146_2():
    input_content = "3\n11 100 1\n13 200 20\n12 300 3"
    expected_output = "11 12 13"
    run_pie_test_case("../p00146.py", input_content, expected_output)


def test_problem_p00146_3():
    input_content = "2\n1 100 1\n2 200 2"
    expected_output = "1 2"
    run_pie_test_case("../p00146.py", input_content, expected_output)
