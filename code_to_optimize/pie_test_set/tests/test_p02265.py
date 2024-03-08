from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02265_0():
    input_content = "7\ninsert 5\ninsert 2\ninsert 3\ninsert 1\ndelete 3\ninsert 6\ndelete 5"
    expected_output = "6 1 2"
    run_pie_test_case("../p02265.py", input_content, expected_output)


def test_problem_p02265_1():
    input_content = "9\ninsert 5\ninsert 2\ninsert 3\ninsert 1\ndelete 3\ninsert 6\ndelete 5\ndeleteFirst\ndeleteLast"
    expected_output = "1"
    run_pie_test_case("../p02265.py", input_content, expected_output)


def test_problem_p02265_2():
    input_content = "7\ninsert 5\ninsert 2\ninsert 3\ninsert 1\ndelete 3\ninsert 6\ndelete 5"
    expected_output = "6 1 2"
    run_pie_test_case("../p02265.py", input_content, expected_output)
