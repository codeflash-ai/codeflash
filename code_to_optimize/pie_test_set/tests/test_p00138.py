from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00138_0():
    input_content = "18 25.46\n16 26.23\n3 23.00\n10 24.79\n5 22.88\n11 23.87\n19 23.90\n1 25.11\n23 23.88\n4 23.46\n7 24.12\n12 22.91\n13 21.99\n14 22.86\n21 23.12\n9 24.09\n17 22.51\n22 23.49\n6 23.02\n20 22.23\n24 21.89\n15 24.14\n8 23.77\n2 23.42"
    expected_output = "5 22.88\n3 23.00\n13 21.99\n14 22.86\n24 21.89\n20 22.23\n17 22.51\n12 22.91"
    run_pie_test_case("../p00138.py", input_content, expected_output)


def test_problem_p00138_1():
    input_content = "18 25.46\n16 26.23\n3 23.00\n10 24.79\n5 22.88\n11 23.87\n19 23.90\n1 25.11\n23 23.88\n4 23.46\n7 24.12\n12 22.91\n13 21.99\n14 22.86\n21 23.12\n9 24.09\n17 22.51\n22 23.49\n6 23.02\n20 22.23\n24 21.89\n15 24.14\n8 23.77\n2 23.42"
    expected_output = "5 22.88\n3 23.00\n13 21.99\n14 22.86\n24 21.89\n20 22.23\n17 22.51\n12 22.91"
    run_pie_test_case("../p00138.py", input_content, expected_output)
