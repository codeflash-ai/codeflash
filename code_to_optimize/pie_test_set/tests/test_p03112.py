from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03112_0():
    input_content = "2 3 4\n100\n600\n400\n900\n1000\n150\n2000\n899\n799"
    expected_output = "350\n1400\n301\n399"
    run_pie_test_case("../p03112.py", input_content, expected_output)


def test_problem_p03112_1():
    input_content = "2 3 4\n100\n600\n400\n900\n1000\n150\n2000\n899\n799"
    expected_output = "350\n1400\n301\n399"
    run_pie_test_case("../p03112.py", input_content, expected_output)


def test_problem_p03112_2():
    input_content = "1 1 3\n1\n10000000000\n2\n9999999999\n5000000000"
    expected_output = "10000000000\n10000000000\n14999999998"
    run_pie_test_case("../p03112.py", input_content, expected_output)
