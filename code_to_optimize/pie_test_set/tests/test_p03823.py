from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03823_0():
    input_content = "5 3 7\n1\n3\n6\n9\n12"
    expected_output = "5"
    run_pie_test_case("../p03823.py", input_content, expected_output)


def test_problem_p03823_1():
    input_content = "5 3 7\n1\n3\n6\n9\n12"
    expected_output = "5"
    run_pie_test_case("../p03823.py", input_content, expected_output)


def test_problem_p03823_2():
    input_content = "3 3 4\n5\n6\n7"
    expected_output = "0"
    run_pie_test_case("../p03823.py", input_content, expected_output)


def test_problem_p03823_3():
    input_content = "8 2 9\n3\n4\n5\n13\n15\n22\n26\n32"
    expected_output = "13"
    run_pie_test_case("../p03823.py", input_content, expected_output)


def test_problem_p03823_4():
    input_content = "7 5 3\n0\n2\n4\n7\n8\n11\n15"
    expected_output = "4"
    run_pie_test_case("../p03823.py", input_content, expected_output)
