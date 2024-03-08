from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03076_0():
    input_content = "29\n20\n7\n35\n120"
    expected_output = "215"
    run_pie_test_case("../p03076.py", input_content, expected_output)


def test_problem_p03076_1():
    input_content = "123\n123\n123\n123\n123"
    expected_output = "643"
    run_pie_test_case("../p03076.py", input_content, expected_output)


def test_problem_p03076_2():
    input_content = "29\n20\n7\n35\n120"
    expected_output = "215"
    run_pie_test_case("../p03076.py", input_content, expected_output)


def test_problem_p03076_3():
    input_content = "101\n86\n119\n108\n57"
    expected_output = "481"
    run_pie_test_case("../p03076.py", input_content, expected_output)
