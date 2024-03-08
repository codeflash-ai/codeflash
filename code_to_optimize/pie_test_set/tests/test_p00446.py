from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00446_0():
    input_content = "5\n1\n7\n9\n6\n10\n10\n8\n7\n14\n18\n4\n11\n3\n17\n5\n19\n0"
    expected_output = "3\n0\n2\n0"
    run_pie_test_case("../p00446.py", input_content, expected_output)


def test_problem_p00446_1():
    input_content = "5\n1\n7\n9\n6\n10\n10\n8\n7\n14\n18\n4\n11\n3\n17\n5\n19\n0"
    expected_output = "3\n0\n2\n0"
    run_pie_test_case("../p00446.py", input_content, expected_output)
