from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02916_0():
    input_content = "3\n3 1 2\n2 5 4\n3 6"
    expected_output = "14"
    run_pie_test_case("../p02916.py", input_content, expected_output)


def test_problem_p02916_1():
    input_content = "2\n1 2\n50 50\n50"
    expected_output = "150"
    run_pie_test_case("../p02916.py", input_content, expected_output)


def test_problem_p02916_2():
    input_content = "4\n2 3 4 1\n13 5 8 24\n45 9 15"
    expected_output = "74"
    run_pie_test_case("../p02916.py", input_content, expected_output)


def test_problem_p02916_3():
    input_content = "3\n3 1 2\n2 5 4\n3 6"
    expected_output = "14"
    run_pie_test_case("../p02916.py", input_content, expected_output)
