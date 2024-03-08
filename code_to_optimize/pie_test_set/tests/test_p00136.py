from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00136_0():
    input_content = "4\n180.3\n168.2\n165.5\n175.3"
    expected_output = "1:\n2:**\n3:\n4:*\n5:*\n6:"
    run_pie_test_case("../p00136.py", input_content, expected_output)


def test_problem_p00136_1():
    input_content = "4\n180.3\n168.2\n165.5\n175.3"
    expected_output = "1:\n2:**\n3:\n4:*\n5:*\n6:"
    run_pie_test_case("../p00136.py", input_content, expected_output)


def test_problem_p00136_2():
    input_content = "21\n179.4\n171.5\n156.6\n173.0\n169.4\n181.2\n172.4\n170.0\n163.6\n165.9\n173.5\n168.2\n162.5\n172.0\n175.1\n172.3\n167.5\n175.9\n186.2\n168.0\n178.6"
    expected_output = "1:***\n2:*****\n3:*******\n4:****\n5:*\n6:*"
    run_pie_test_case("../p00136.py", input_content, expected_output)
