from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03994_0():
    input_content = "xyz\n4"
    expected_output = "aya"
    run_pie_test_case("../p03994.py", input_content, expected_output)


def test_problem_p03994_1():
    input_content = "codefestival\n100"
    expected_output = "aaaafeaaivap"
    run_pie_test_case("../p03994.py", input_content, expected_output)


def test_problem_p03994_2():
    input_content = "xyz\n4"
    expected_output = "aya"
    run_pie_test_case("../p03994.py", input_content, expected_output)


def test_problem_p03994_3():
    input_content = "a\n25"
    expected_output = "z"
    run_pie_test_case("../p03994.py", input_content, expected_output)
