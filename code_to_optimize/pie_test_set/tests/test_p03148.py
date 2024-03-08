from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03148_0():
    input_content = "5 3\n1 9\n1 7\n2 6\n2 5\n3 1"
    expected_output = "26"
    run_pie_test_case("../p03148.py", input_content, expected_output)


def test_problem_p03148_1():
    input_content = (
        "6 5\n5 1000000000\n2 990000000\n3 980000000\n6 970000000\n6 960000000\n4 950000000"
    )
    expected_output = "4900000016"
    run_pie_test_case("../p03148.py", input_content, expected_output)


def test_problem_p03148_2():
    input_content = "7 4\n1 1\n2 1\n3 1\n4 6\n4 5\n4 5\n4 5"
    expected_output = "25"
    run_pie_test_case("../p03148.py", input_content, expected_output)


def test_problem_p03148_3():
    input_content = "5 3\n1 9\n1 7\n2 6\n2 5\n3 1"
    expected_output = "26"
    run_pie_test_case("../p03148.py", input_content, expected_output)
