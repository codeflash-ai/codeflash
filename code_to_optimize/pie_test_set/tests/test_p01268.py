from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01268_0():
    input_content = "0 55\n0 1\n0 2\n0 3\n10 1\n10 2\n10 3\n10 4\n10 5\n10 6\n11 1\n11 2\n11 3\n100000 100\n-1 -1"
    expected_output = "42\n4\n5\n6\n22\n24\n26\n28\n30\n30\n26\n30\n32\n200274"
    run_pie_test_case("../p01268.py", input_content, expected_output)


def test_problem_p01268_1():
    input_content = "0 55\n0 1\n0 2\n0 3\n10 1\n10 2\n10 3\n10 4\n10 5\n10 6\n11 1\n11 2\n11 3\n100000 100\n-1 -1"
    expected_output = "42\n4\n5\n6\n22\n24\n26\n28\n30\n30\n26\n30\n32\n200274"
    run_pie_test_case("../p01268.py", input_content, expected_output)
