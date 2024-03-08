from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03016_0():
    input_content = "5 3 4 10007"
    expected_output = "5563"
    run_pie_test_case("../p03016.py", input_content, expected_output)


def test_problem_p03016_1():
    input_content = "4 8 1 1000000"
    expected_output = "891011"
    run_pie_test_case("../p03016.py", input_content, expected_output)


def test_problem_p03016_2():
    input_content = "107 10000000000007 1000000000000007 998244353"
    expected_output = "39122908"
    run_pie_test_case("../p03016.py", input_content, expected_output)


def test_problem_p03016_3():
    input_content = "5 3 4 10007"
    expected_output = "5563"
    run_pie_test_case("../p03016.py", input_content, expected_output)
