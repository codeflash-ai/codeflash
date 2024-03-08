from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02605_0():
    input_content = "2\n11 1 U\n11 47 D"
    expected_output = "230"
    run_pie_test_case("../p02605.py", input_content, expected_output)


def test_problem_p02605_1():
    input_content = (
        "8\n168 224 U\n130 175 R\n111 198 D\n121 188 L\n201 116 U\n112 121 R\n145 239 D\n185 107 L"
    )
    expected_output = "100"
    run_pie_test_case("../p02605.py", input_content, expected_output)


def test_problem_p02605_2():
    input_content = "4\n20 30 U\n30 20 R\n20 10 D\n10 20 L"
    expected_output = "SAFE"
    run_pie_test_case("../p02605.py", input_content, expected_output)


def test_problem_p02605_3():
    input_content = "2\n11 1 U\n11 47 D"
    expected_output = "230"
    run_pie_test_case("../p02605.py", input_content, expected_output)
