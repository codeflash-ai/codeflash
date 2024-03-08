from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03140_0():
    input_content = "4\nwest\neast\nwait"
    expected_output = "3"
    run_pie_test_case("../p03140.py", input_content, expected_output)


def test_problem_p03140_1():
    input_content = "9\ndifferent\ndifferent\ndifferent"
    expected_output = "0"
    run_pie_test_case("../p03140.py", input_content, expected_output)


def test_problem_p03140_2():
    input_content = "4\nwest\neast\nwait"
    expected_output = "3"
    run_pie_test_case("../p03140.py", input_content, expected_output)


def test_problem_p03140_3():
    input_content = "7\nzenkoku\ntouitsu\nprogram"
    expected_output = "13"
    run_pie_test_case("../p03140.py", input_content, expected_output)
