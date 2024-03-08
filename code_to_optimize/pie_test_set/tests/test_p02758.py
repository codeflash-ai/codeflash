from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02758_0():
    input_content = "2\n1 5\n3 3"
    expected_output = "3"
    run_pie_test_case("../p02758.py", input_content, expected_output)


def test_problem_p02758_1():
    input_content = "2\n1 5\n3 3"
    expected_output = "3"
    run_pie_test_case("../p02758.py", input_content, expected_output)


def test_problem_p02758_2():
    input_content = "3\n6 5\n-1 10\n3 3"
    expected_output = "5"
    run_pie_test_case("../p02758.py", input_content, expected_output)


def test_problem_p02758_3():
    input_content = "20\n-8 1\n26 4\n0 5\n9 1\n19 4\n22 20\n28 27\n11 8\n-3 20\n-25 17\n10 4\n-18 27\n24 28\n-11 19\n2 27\n-2 18\n-1 12\n-24 29\n31 29\n29 7"
    expected_output = "110"
    run_pie_test_case("../p02758.py", input_content, expected_output)


def test_problem_p02758_4():
    input_content = "4\n7 10\n-10 3\n4 3\n-4 3"
    expected_output = "16"
    run_pie_test_case("../p02758.py", input_content, expected_output)
