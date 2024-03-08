from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00159_0():
    input_content = "6\n1 165 66\n2 178 60\n3 180 72\n4 160 65\n5 185 62\n6 182 62\n3\n3 160 65\n2 180 70\n1 170 75\n0"
    expected_output = "3\n2"
    run_pie_test_case("../p00159.py", input_content, expected_output)


def test_problem_p00159_1():
    input_content = "6\n1 165 66\n2 178 60\n3 180 72\n4 160 65\n5 185 62\n6 182 62\n3\n3 160 65\n2 180 70\n1 170 75\n0"
    expected_output = "3\n2"
    run_pie_test_case("../p00159.py", input_content, expected_output)
