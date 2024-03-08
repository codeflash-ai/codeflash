from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03326_0():
    input_content = "5 3\n3 1 4\n1 5 9\n2 6 5\n3 5 8\n9 7 9"
    expected_output = "56"
    run_pie_test_case("../p03326.py", input_content, expected_output)


def test_problem_p03326_1():
    input_content = "5 3\n1 -2 3\n-4 5 -6\n7 -8 -9\n-10 11 -12\n13 -14 15"
    expected_output = "54"
    run_pie_test_case("../p03326.py", input_content, expected_output)


def test_problem_p03326_2():
    input_content = "10 5\n10 -80 21\n23 8 38\n-94 28 11\n-26 -2 18\n-69 72 79\n-26 -86 -54\n-72 -50 59\n21 65 -32\n40 -94 87\n-62 18 82"
    expected_output = "638"
    run_pie_test_case("../p03326.py", input_content, expected_output)


def test_problem_p03326_3():
    input_content = "5 3\n3 1 4\n1 5 9\n2 6 5\n3 5 8\n9 7 9"
    expected_output = "56"
    run_pie_test_case("../p03326.py", input_content, expected_output)


def test_problem_p03326_4():
    input_content = "3 2\n2000000000 -9000000000 4000000000\n7000000000 -5000000000 3000000000\n6000000000 -1000000000 8000000000"
    expected_output = "30000000000"
    run_pie_test_case("../p03326.py", input_content, expected_output)
