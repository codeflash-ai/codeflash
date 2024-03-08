from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02989_0():
    input_content = "6\n9 1 4 4 6 7"
    expected_output = "2"
    run_pie_test_case("../p02989.py", input_content, expected_output)


def test_problem_p02989_1():
    input_content = "8\n9 1 14 5 5 4 4 14"
    expected_output = "0"
    run_pie_test_case("../p02989.py", input_content, expected_output)


def test_problem_p02989_2():
    input_content = "6\n9 1 4 4 6 7"
    expected_output = "2"
    run_pie_test_case("../p02989.py", input_content, expected_output)


def test_problem_p02989_3():
    input_content = (
        "14\n99592 10342 29105 78532 83018 11639 92015 77204 30914 21912 34519 80835 100000 1"
    )
    expected_output = "42685"
    run_pie_test_case("../p02989.py", input_content, expected_output)
