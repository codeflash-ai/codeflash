from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03162_0():
    input_content = "3\n10 40 70\n20 50 80\n30 60 90"
    expected_output = "210"
    run_pie_test_case("../p03162.py", input_content, expected_output)


def test_problem_p03162_1():
    input_content = "3\n10 40 70\n20 50 80\n30 60 90"
    expected_output = "210"
    run_pie_test_case("../p03162.py", input_content, expected_output)


def test_problem_p03162_2():
    input_content = "1\n100 10 1"
    expected_output = "100"
    run_pie_test_case("../p03162.py", input_content, expected_output)


def test_problem_p03162_3():
    input_content = "7\n6 7 8\n8 8 3\n2 5 2\n7 8 6\n4 6 8\n2 3 4\n7 5 1"
    expected_output = "46"
    run_pie_test_case("../p03162.py", input_content, expected_output)
