from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02738_0():
    input_content = "1 998244353"
    expected_output = "6"
    run_pie_test_case("../p02738.py", input_content, expected_output)


def test_problem_p02738_1():
    input_content = "314 1000000007"
    expected_output = "182908545"
    run_pie_test_case("../p02738.py", input_content, expected_output)


def test_problem_p02738_2():
    input_content = "1 998244353"
    expected_output = "6"
    run_pie_test_case("../p02738.py", input_content, expected_output)


def test_problem_p02738_3():
    input_content = "2 998244353"
    expected_output = "261"
    run_pie_test_case("../p02738.py", input_content, expected_output)
