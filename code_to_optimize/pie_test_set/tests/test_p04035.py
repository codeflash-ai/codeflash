from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04035_0():
    input_content = "3 50\n30 20 10"
    expected_output = "Possible\n2\n1"
    run_pie_test_case("../p04035.py", input_content, expected_output)


def test_problem_p04035_1():
    input_content = "3 50\n30 20 10"
    expected_output = "Possible\n2\n1"
    run_pie_test_case("../p04035.py", input_content, expected_output)


def test_problem_p04035_2():
    input_content = "5 50\n10 20 30 40 50"
    expected_output = "Possible\n1\n2\n3\n4"
    run_pie_test_case("../p04035.py", input_content, expected_output)


def test_problem_p04035_3():
    input_content = "2 21\n10 10"
    expected_output = "Impossible"
    run_pie_test_case("../p04035.py", input_content, expected_output)
