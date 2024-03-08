from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02891_0():
    input_content = "issii\n2"
    expected_output = "4"
    run_pie_test_case("../p02891.py", input_content, expected_output)


def test_problem_p02891_1():
    input_content = "cooooooooonteeeeeeeeeest\n999993333"
    expected_output = "8999939997"
    run_pie_test_case("../p02891.py", input_content, expected_output)


def test_problem_p02891_2():
    input_content = "issii\n2"
    expected_output = "4"
    run_pie_test_case("../p02891.py", input_content, expected_output)


def test_problem_p02891_3():
    input_content = "qq\n81"
    expected_output = "81"
    run_pie_test_case("../p02891.py", input_content, expected_output)
