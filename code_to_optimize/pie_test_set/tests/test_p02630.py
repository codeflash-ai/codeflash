from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02630_0():
    input_content = "4\n1 2 3 4\n3\n1 2\n3 4\n2 4"
    expected_output = "11\n12\n16"
    run_pie_test_case("../p02630.py", input_content, expected_output)


def test_problem_p02630_1():
    input_content = "2\n1 2\n3\n1 100\n2 100\n100 1000"
    expected_output = "102\n200\n2000"
    run_pie_test_case("../p02630.py", input_content, expected_output)


def test_problem_p02630_2():
    input_content = "4\n1 2 3 4\n3\n1 2\n3 4\n2 4"
    expected_output = "11\n12\n16"
    run_pie_test_case("../p02630.py", input_content, expected_output)


def test_problem_p02630_3():
    input_content = "4\n1 1 1 1\n3\n1 2\n2 1\n3 5"
    expected_output = "8\n4\n4"
    run_pie_test_case("../p02630.py", input_content, expected_output)
