from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03733_0():
    input_content = "2 4\n0 3"
    expected_output = "7"
    run_pie_test_case("../p03733.py", input_content, expected_output)


def test_problem_p03733_1():
    input_content = "9 10\n0 3 5 7 100 110 200 300 311"
    expected_output = "67"
    run_pie_test_case("../p03733.py", input_content, expected_output)


def test_problem_p03733_2():
    input_content = "2 4\n0 3"
    expected_output = "7"
    run_pie_test_case("../p03733.py", input_content, expected_output)


def test_problem_p03733_3():
    input_content = "1 1\n0"
    expected_output = "1"
    run_pie_test_case("../p03733.py", input_content, expected_output)


def test_problem_p03733_4():
    input_content = "2 4\n0 5"
    expected_output = "8"
    run_pie_test_case("../p03733.py", input_content, expected_output)


def test_problem_p03733_5():
    input_content = "4 1000000000\n0 1000 1000000 1000000000"
    expected_output = "2000000000"
    run_pie_test_case("../p03733.py", input_content, expected_output)
