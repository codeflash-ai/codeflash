from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03765_0():
    input_content = "BBBAAAABA\nBBBBA\n4\n7 9 2 5\n7 9 1 4\n1 7 2 5\n1 7 2 4"
    expected_output = "YES\nNO\nYES\nNO"
    run_pie_test_case("../p03765.py", input_content, expected_output)


def test_problem_p03765_1():
    input_content = "AAAAABBBBAAABBBBAAAA\nBBBBAAABBBBBBAAAAABB\n10\n2 15 2 13\n2 13 6 16\n1 13 2 20\n4 20 3 20\n1 18 9 19\n2 14 1 11\n3 20 3 15\n6 16 1 17\n4 18 8 20\n7 20 3 14"
    expected_output = "YES\nYES\nYES\nYES\nYES\nYES\nNO\nNO\nNO\nNO"
    run_pie_test_case("../p03765.py", input_content, expected_output)


def test_problem_p03765_2():
    input_content = "BBBAAAABA\nBBBBA\n4\n7 9 2 5\n7 9 1 4\n1 7 2 5\n1 7 2 4"
    expected_output = "YES\nNO\nYES\nNO"
    run_pie_test_case("../p03765.py", input_content, expected_output)
