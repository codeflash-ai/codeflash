from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03399_0():
    input_content = "600\n300\n220\n420"
    expected_output = "520"
    run_pie_test_case("../p03399.py", input_content, expected_output)


def test_problem_p03399_1():
    input_content = "549\n817\n715\n603"
    expected_output = "1152"
    run_pie_test_case("../p03399.py", input_content, expected_output)


def test_problem_p03399_2():
    input_content = "555\n555\n400\n200"
    expected_output = "755"
    run_pie_test_case("../p03399.py", input_content, expected_output)


def test_problem_p03399_3():
    input_content = "600\n300\n220\n420"
    expected_output = "520"
    run_pie_test_case("../p03399.py", input_content, expected_output)
