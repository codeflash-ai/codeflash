from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01105_0():
    input_content = "0\n(a*(1*b))\n(1^a)\n(-(-a*-b)*a)\n(a^(b^(c^d)))\n."
    expected_output = "1\n5\n2\n1\n13"
    run_pie_test_case("../p01105.py", input_content, expected_output)


def test_problem_p01105_1():
    input_content = "0\n(a*(1*b))\n(1^a)\n(-(-a*-b)*a)\n(a^(b^(c^d)))\n."
    expected_output = "1\n5\n2\n1\n13"
    run_pie_test_case("../p01105.py", input_content, expected_output)
