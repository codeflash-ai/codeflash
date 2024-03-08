from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03313_0():
    input_content = "2\n1 2 3 1"
    expected_output = "3\n4\n5"
    run_pie_test_case("../p03313.py", input_content, expected_output)


def test_problem_p03313_1():
    input_content = "2\n1 2 3 1"
    expected_output = "3\n4\n5"
    run_pie_test_case("../p03313.py", input_content, expected_output)


def test_problem_p03313_2():
    input_content = "3\n10 71 84 33 6 47 23 25"
    expected_output = "81\n94\n155\n155\n155\n155\n155"
    run_pie_test_case("../p03313.py", input_content, expected_output)


def test_problem_p03313_3():
    input_content = "4\n75 26 45 72 81 47 97 97 2 2 25 82 84 17 56 32"
    expected_output = "101\n120\n147\n156\n156\n178\n194\n194\n194\n194\n194\n194\n194\n194\n194"
    run_pie_test_case("../p03313.py", input_content, expected_output)
