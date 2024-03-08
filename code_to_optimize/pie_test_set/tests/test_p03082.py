from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03082_0():
    input_content = "2 19\n3 7"
    expected_output = "3"
    run_pie_test_case("../p03082.py", input_content, expected_output)


def test_problem_p03082_1():
    input_content = "5 82\n22 11 6 5 13"
    expected_output = "288"
    run_pie_test_case("../p03082.py", input_content, expected_output)


def test_problem_p03082_2():
    input_content = "10 100000\n50000 50001 50002 50003 50004 50005 50006 50007 50008 50009"
    expected_output = "279669259"
    run_pie_test_case("../p03082.py", input_content, expected_output)


def test_problem_p03082_3():
    input_content = "2 19\n3 7"
    expected_output = "3"
    run_pie_test_case("../p03082.py", input_content, expected_output)
