from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03290_0():
    input_content = "2 700\n3 500\n5 800"
    expected_output = "3"
    run_pie_test_case("../p03290.py", input_content, expected_output)


def test_problem_p03290_1():
    input_content = "5 25000\n20 1000\n40 1000\n50 1000\n30 1000\n1 1000"
    expected_output = "66"
    run_pie_test_case("../p03290.py", input_content, expected_output)


def test_problem_p03290_2():
    input_content = "2 400\n3 500\n5 800"
    expected_output = "2"
    run_pie_test_case("../p03290.py", input_content, expected_output)


def test_problem_p03290_3():
    input_content = "2 700\n3 500\n5 800"
    expected_output = "3"
    run_pie_test_case("../p03290.py", input_content, expected_output)


def test_problem_p03290_4():
    input_content = "2 2000\n3 500\n5 800"
    expected_output = "7"
    run_pie_test_case("../p03290.py", input_content, expected_output)
