from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00184_0():
    input_content = "8\n71\n34\n65\n11\n41\n39\n6\n5\n4\n67\n81\n78\n65\n0"
    expected_output = "2\n1\n0\n2\n1\n0\n2\n0\n0\n0\n0\n0\n0\n4"
    run_pie_test_case("../p00184.py", input_content, expected_output)


def test_problem_p00184_1():
    input_content = "8\n71\n34\n65\n11\n41\n39\n6\n5\n4\n67\n81\n78\n65\n0"
    expected_output = "2\n1\n0\n2\n1\n0\n2\n0\n0\n0\n0\n0\n0\n4"
    run_pie_test_case("../p00184.py", input_content, expected_output)
