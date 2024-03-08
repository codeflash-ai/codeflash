from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02815_0():
    input_content = "1\n1000000000"
    expected_output = "999999993"
    run_pie_test_case("../p02815.py", input_content, expected_output)


def test_problem_p02815_1():
    input_content = "2\n5 8"
    expected_output = "124"
    run_pie_test_case("../p02815.py", input_content, expected_output)


def test_problem_p02815_2():
    input_content = "1\n1000000000"
    expected_output = "999999993"
    run_pie_test_case("../p02815.py", input_content, expected_output)


def test_problem_p02815_3():
    input_content = "5\n52 67 72 25 79"
    expected_output = "269312"
    run_pie_test_case("../p02815.py", input_content, expected_output)
