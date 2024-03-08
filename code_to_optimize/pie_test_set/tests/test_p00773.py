from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00773_0():
    input_content = "5 8 105\n8 5 105\n1 2 24\n99 98 24\n12 13 26\n1 22 23\n1 13 201\n13 16 112\n2 24 50\n1 82 61\n1 84 125\n1 99 999\n99 1 999\n98 99 999\n1 99 11\n99 1 12\n0 0 0"
    expected_output = "109\n103\n24\n24\n26\n27\n225\n116\n62\n111\n230\n1972\n508\n1004\n20\n7"
    run_pie_test_case("../p00773.py", input_content, expected_output)


def test_problem_p00773_1():
    input_content = "5 8 105\n8 5 105\n1 2 24\n99 98 24\n12 13 26\n1 22 23\n1 13 201\n13 16 112\n2 24 50\n1 82 61\n1 84 125\n1 99 999\n99 1 999\n98 99 999\n1 99 11\n99 1 12\n0 0 0"
    expected_output = "109\n103\n24\n24\n26\n27\n225\n116\n62\n111\n230\n1972\n508\n1004\n20\n7"
    run_pie_test_case("../p00773.py", input_content, expected_output)
