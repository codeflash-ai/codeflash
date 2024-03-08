from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03510_0():
    input_content = "5\n10 20\n40 50\n60 30\n70 40\n90 10"
    expected_output = "90"
    run_pie_test_case("../p03510.py", input_content, expected_output)


def test_problem_p03510_1():
    input_content = "4\n1 100\n3 200\n999999999999999 150\n1000000000000000 150"
    expected_output = "299"
    run_pie_test_case("../p03510.py", input_content, expected_output)


def test_problem_p03510_2():
    input_content = "5\n10 2\n40 5\n60 3\n70 4\n90 1"
    expected_output = "5"
    run_pie_test_case("../p03510.py", input_content, expected_output)


def test_problem_p03510_3():
    input_content = "5\n10 20\n40 50\n60 30\n70 40\n90 10"
    expected_output = "90"
    run_pie_test_case("../p03510.py", input_content, expected_output)
