from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00707_0():
    input_content = "7 4\n9R2A993\n0E314A0\n8A900DE\n820R037\n6 7\nJH03HE\nID7722\n0DA1AH\n30C9G5\n99971A\nCA7EAI\nAHLBEM\n20 2\nA1234567891234CBDEGH\nBDEDF908034265091499\n0 0"
    expected_output = "23900037\n771971\n12345908034265091499"
    run_pie_test_case("../p00707.py", input_content, expected_output)


def test_problem_p00707_1():
    input_content = "7 4\n9R2A993\n0E314A0\n8A900DE\n820R037\n6 7\nJH03HE\nID7722\n0DA1AH\n30C9G5\n99971A\nCA7EAI\nAHLBEM\n20 2\nA1234567891234CBDEGH\nBDEDF908034265091499\n0 0"
    expected_output = "23900037\n771971\n12345908034265091499"
    run_pie_test_case("../p00707.py", input_content, expected_output)
