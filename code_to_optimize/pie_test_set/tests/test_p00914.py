from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00914_0():
    input_content = (
        "9 3 23\n9 3 22\n10 3 28\n16 10 107\n20 8 102\n20 10 105\n20 10 155\n3 4 3\n4 2 11\n0 0 0"
    )
    expected_output = "1\n2\n0\n20\n1542\n5448\n1\n0\n0"
    run_pie_test_case("../p00914.py", input_content, expected_output)


def test_problem_p00914_1():
    input_content = (
        "9 3 23\n9 3 22\n10 3 28\n16 10 107\n20 8 102\n20 10 105\n20 10 155\n3 4 3\n4 2 11\n0 0 0"
    )
    expected_output = "1\n2\n0\n20\n1542\n5448\n1\n0\n0"
    run_pie_test_case("../p00914.py", input_content, expected_output)
