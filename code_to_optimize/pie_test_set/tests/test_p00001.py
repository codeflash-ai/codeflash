from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00001_0():
    input_content = "1819\n2003\n876\n2840\n1723\n1673\n3776\n2848\n1592\n922"
    expected_output = "3776\n2848\n2840"
    run_pie_test_case("../p00001.py", input_content, expected_output)


def test_problem_p00001_1():
    input_content = "1819\n2003\n876\n2840\n1723\n1673\n3776\n2848\n1592\n922"
    expected_output = "3776\n2848\n2840"
    run_pie_test_case("../p00001.py", input_content, expected_output)


def test_problem_p00001_2():
    input_content = "100\n200\n300\n400\n500\n600\n700\n800\n900\n900"
    expected_output = "900\n900\n800"
    run_pie_test_case("../p00001.py", input_content, expected_output)
