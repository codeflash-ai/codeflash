from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03409_0():
    input_content = "3\n2 0\n3 1\n1 3\n4 2\n0 4\n5 5"
    expected_output = "2"
    run_pie_test_case("../p03409.py", input_content, expected_output)


def test_problem_p03409_1():
    input_content = "2\n2 2\n3 3\n0 0\n1 1"
    expected_output = "0"
    run_pie_test_case("../p03409.py", input_content, expected_output)


def test_problem_p03409_2():
    input_content = "5\n0 0\n7 3\n2 2\n4 8\n1 6\n8 5\n6 9\n5 4\n9 1\n3 7"
    expected_output = "5"
    run_pie_test_case("../p03409.py", input_content, expected_output)


def test_problem_p03409_3():
    input_content = "5\n0 0\n1 1\n5 5\n6 6\n7 7\n2 2\n3 3\n4 4\n8 8\n9 9"
    expected_output = "4"
    run_pie_test_case("../p03409.py", input_content, expected_output)


def test_problem_p03409_4():
    input_content = "3\n0 0\n1 1\n5 2\n2 3\n3 4\n4 5"
    expected_output = "2"
    run_pie_test_case("../p03409.py", input_content, expected_output)


def test_problem_p03409_5():
    input_content = "3\n2 0\n3 1\n1 3\n4 2\n0 4\n5 5"
    expected_output = "2"
    run_pie_test_case("../p03409.py", input_content, expected_output)
