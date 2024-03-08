from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03566_0():
    input_content = "1\n100\n30"
    expected_output = "2100.000000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)


def test_problem_p03566_1():
    input_content = "1\n9\n10"
    expected_output = "20.250000000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)


def test_problem_p03566_2():
    input_content = "3\n12 14 2\n6 2 7"
    expected_output = "76.000000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)


def test_problem_p03566_3():
    input_content = "2\n60 50\n34 38"
    expected_output = "2632.000000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)


def test_problem_p03566_4():
    input_content = "1\n100\n30"
    expected_output = "2100.000000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)


def test_problem_p03566_5():
    input_content = "10\n64 55 27 35 76 119 7 18 49 100\n29 19 31 39 27 48 41 87 55 70"
    expected_output = "20291.000000000000"
    run_pie_test_case("../p03566.py", input_content, expected_output)
