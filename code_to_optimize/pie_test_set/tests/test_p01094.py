from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01094_0():
    input_content = "1\nA\n4\nA A B B\n5\nL M N L N\n6\nK K K K K K\n6\nX X X Y Z X\n10\nA A A B A C A C C B\n10\nU U U U U V V W W W\n0"
    expected_output = "A 1\nTIE\nTIE\nK 4\nX 5\nA 7\nU 8"
    run_pie_test_case("../p01094.py", input_content, expected_output)


def test_problem_p01094_1():
    input_content = "1\nA\n4\nA A B B\n5\nL M N L N\n6\nK K K K K K\n6\nX X X Y Z X\n10\nA A A B A C A C C B\n10\nU U U U U V V W W W\n0"
    expected_output = "A 1\nTIE\nTIE\nK 4\nX 5\nA 7\nU 8"
    run_pie_test_case("../p01094.py", input_content, expected_output)
