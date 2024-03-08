from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03251_0():
    input_content = "3 2 10 20\n8 15 13\n16 22"
    expected_output = "No War"
    run_pie_test_case("../p03251.py", input_content, expected_output)


def test_problem_p03251_1():
    input_content = "3 2 10 20\n8 15 13\n16 22"
    expected_output = "No War"
    run_pie_test_case("../p03251.py", input_content, expected_output)


def test_problem_p03251_2():
    input_content = "4 2 -48 -1\n-20 -35 -91 -23\n-22 66"
    expected_output = "War"
    run_pie_test_case("../p03251.py", input_content, expected_output)


def test_problem_p03251_3():
    input_content = "5 3 6 8\n-10 3 1 5 -100\n100 6 14"
    expected_output = "War"
    run_pie_test_case("../p03251.py", input_content, expected_output)
