from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03303_0():
    input_content = "abcdefgh\n3"
    expected_output = "adg"
    run_pie_test_case("../p03303.py", input_content, expected_output)


def test_problem_p03303_1():
    input_content = "lllll\n1"
    expected_output = "lllll"
    run_pie_test_case("../p03303.py", input_content, expected_output)


def test_problem_p03303_2():
    input_content = "abcdefgh\n3"
    expected_output = "adg"
    run_pie_test_case("../p03303.py", input_content, expected_output)


def test_problem_p03303_3():
    input_content = "souuundhound\n2"
    expected_output = "suudon"
    run_pie_test_case("../p03303.py", input_content, expected_output)
