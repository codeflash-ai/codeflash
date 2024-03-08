from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03413_0():
    input_content = "5\n1 4 3 7 5"
    expected_output = "11\n3\n1\n4\n2"
    run_pie_test_case("../p03413.py", input_content, expected_output)


def test_problem_p03413_1():
    input_content = "5\n1 4 3 7 5"
    expected_output = "11\n3\n1\n4\n2"
    run_pie_test_case("../p03413.py", input_content, expected_output)


def test_problem_p03413_2():
    input_content = "6\n-1 -2 -3 1 2 3"
    expected_output = "4\n3\n2\n1\n2"
    run_pie_test_case("../p03413.py", input_content, expected_output)


def test_problem_p03413_3():
    input_content = "4\n100 100 -1 100"
    expected_output = "200\n2\n3\n1"
    run_pie_test_case("../p03413.py", input_content, expected_output)


def test_problem_p03413_4():
    input_content = "9\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "5000000000\n4\n2\n2\n2\n2"
    run_pie_test_case("../p03413.py", input_content, expected_output)
