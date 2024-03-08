from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00721_0():
    input_content = "7 5\n.......\n.o...*.\n.......\n.*...*.\n.......\n15 13\n.......x.......\n...o...x....*..\n.......x.......\n.......x.......\n.......x.......\n...............\nxxxxx.....xxxxx\n...............\n.......x.......\n.......x.......\n.......x.......\n..*....x....*..\n.......x.......\n10 10\n..........\n..o.......\n..........\n..........\n..........\n.....xxxxx\n.....x....\n.....x.*..\n.....x....\n.....x....\n0 0"
    expected_output = "8\n49\n-1"
    run_pie_test_case("../p00721.py", input_content, expected_output)


def test_problem_p00721_1():
    input_content = "7 5\n.......\n.o...*.\n.......\n.*...*.\n.......\n15 13\n.......x.......\n...o...x....*..\n.......x.......\n.......x.......\n.......x.......\n...............\nxxxxx.....xxxxx\n...............\n.......x.......\n.......x.......\n.......x.......\n..*....x....*..\n.......x.......\n10 10\n..........\n..o.......\n..........\n..........\n..........\n.....xxxxx\n.....x....\n.....x.*..\n.....x....\n.....x....\n0 0"
    expected_output = "8\n49\n-1"
    run_pie_test_case("../p00721.py", input_content, expected_output)
