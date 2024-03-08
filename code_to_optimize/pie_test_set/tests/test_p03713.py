from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03713_0():
    input_content = "3 5"
    expected_output = "0"
    run_pie_test_case("../p03713.py", input_content, expected_output)


def test_problem_p03713_1():
    input_content = "3 5"
    expected_output = "0"
    run_pie_test_case("../p03713.py", input_content, expected_output)


def test_problem_p03713_2():
    input_content = "100000 100000"
    expected_output = "50000"
    run_pie_test_case("../p03713.py", input_content, expected_output)


def test_problem_p03713_3():
    input_content = "100000 2"
    expected_output = "1"
    run_pie_test_case("../p03713.py", input_content, expected_output)


def test_problem_p03713_4():
    input_content = "5 5"
    expected_output = "4"
    run_pie_test_case("../p03713.py", input_content, expected_output)


def test_problem_p03713_5():
    input_content = "4 5"
    expected_output = "2"
    run_pie_test_case("../p03713.py", input_content, expected_output)
