from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03475_0():
    input_content = "3\n6 5 1\n1 10 1"
    expected_output = "12\n11\n0"
    run_pie_test_case("../p03475.py", input_content, expected_output)


def test_problem_p03475_1():
    input_content = "3\n6 5 1\n1 10 1"
    expected_output = "12\n11\n0"
    run_pie_test_case("../p03475.py", input_content, expected_output)


def test_problem_p03475_2():
    input_content = "4\n12 13 1\n44 17 17\n66 4096 64"
    expected_output = "4162\n4162\n4162\n0"
    run_pie_test_case("../p03475.py", input_content, expected_output)


def test_problem_p03475_3():
    input_content = "4\n12 24 6\n52 16 4\n99 2 2"
    expected_output = "187\n167\n101\n0"
    run_pie_test_case("../p03475.py", input_content, expected_output)
