from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02788_0():
    input_content = "3 3 2\n1 2\n5 4\n9 2"
    expected_output = "2"
    run_pie_test_case("../p02788.py", input_content, expected_output)


def test_problem_p02788_1():
    input_content = "9 4 1\n1 5\n2 4\n3 3\n4 2\n5 1\n6 2\n7 3\n8 4\n9 5"
    expected_output = "5"
    run_pie_test_case("../p02788.py", input_content, expected_output)


def test_problem_p02788_2():
    input_content = "3 3 2\n1 2\n5 4\n9 2"
    expected_output = "2"
    run_pie_test_case("../p02788.py", input_content, expected_output)


def test_problem_p02788_3():
    input_content = "3 0 1\n300000000 1000000000\n100000000 1000000000\n200000000 1000000000"
    expected_output = "3000000000"
    run_pie_test_case("../p02788.py", input_content, expected_output)
