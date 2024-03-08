from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02821_0():
    input_content = "5 3\n10 14 19 34 33"
    expected_output = "202"
    run_pie_test_case("../p02821.py", input_content, expected_output)


def test_problem_p02821_1():
    input_content = "5 3\n10 14 19 34 33"
    expected_output = "202"
    run_pie_test_case("../p02821.py", input_content, expected_output)


def test_problem_p02821_2():
    input_content = "9 14\n1 3 5 110 24 21 34 5 3"
    expected_output = "1837"
    run_pie_test_case("../p02821.py", input_content, expected_output)


def test_problem_p02821_3():
    input_content = "9 73\n67597 52981 5828 66249 75177 64141 40773 79105 16076"
    expected_output = "8128170"
    run_pie_test_case("../p02821.py", input_content, expected_output)
