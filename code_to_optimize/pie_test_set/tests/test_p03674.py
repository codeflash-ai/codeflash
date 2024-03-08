from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03674_0():
    input_content = "3\n1 2 1 3"
    expected_output = "3\n5\n4\n1"
    run_pie_test_case("../p03674.py", input_content, expected_output)


def test_problem_p03674_1():
    input_content = "1\n1 1"
    expected_output = "1\n1"
    run_pie_test_case("../p03674.py", input_content, expected_output)


def test_problem_p03674_2():
    input_content = "32\n29 19 7 10 26 32 27 4 11 20 2 8 16 23 5 14 6 12 17 22 18 30 28 24 15 1 25 3 13 21 19 31 9"
    expected_output = "32\n525\n5453\n40919\n237336\n1107568\n4272048\n13884156\n38567100\n92561040\n193536720\n354817320\n573166440\n818809200\n37158313\n166803103\n166803103\n37158313\n818809200\n573166440\n354817320\n193536720\n92561040\n38567100\n13884156\n4272048\n1107568\n237336\n40920\n5456\n528\n33\n1"
    run_pie_test_case("../p03674.py", input_content, expected_output)


def test_problem_p03674_3():
    input_content = "3\n1 2 1 3"
    expected_output = "3\n5\n4\n1"
    run_pie_test_case("../p03674.py", input_content, expected_output)
