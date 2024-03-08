from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02586_0():
    input_content = "2 2 3\n1 1 3\n2 1 4\n1 2 5"
    expected_output = "8"
    run_pie_test_case("../p02586.py", input_content, expected_output)


def test_problem_p02586_1():
    input_content = "2 5 5\n1 1 3\n2 4 20\n1 2 1\n1 3 4\n1 4 2"
    expected_output = "29"
    run_pie_test_case("../p02586.py", input_content, expected_output)


def test_problem_p02586_2():
    input_content = "2 2 3\n1 1 3\n2 1 4\n1 2 5"
    expected_output = "8"
    run_pie_test_case("../p02586.py", input_content, expected_output)


def test_problem_p02586_3():
    input_content = (
        "4 5 10\n2 5 12\n1 5 12\n2 3 15\n1 2 20\n1 1 28\n2 4 26\n3 2 27\n4 5 21\n3 5 10\n1 3 10"
    )
    expected_output = "142"
    run_pie_test_case("../p02586.py", input_content, expected_output)
