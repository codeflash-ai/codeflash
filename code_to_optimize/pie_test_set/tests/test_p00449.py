from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00449_0():
    input_content = "3 8\n1 3 1 10\n0 2 3\n1 2 3 20\n1 1 2 5\n0 3 2\n1 1 3 7\n1 2 1 9\n0 2 3\n5 16\n1 1 2 343750\n1 1 3 3343\n1 1 4 347392\n1 1 5 5497\n1 2 3 123394\n1 2 4 545492\n1 2 5 458\n1 3 4 343983\n1 3 5 843468\n1 4 5 15934\n0 2 1\n0 4 1\n0 3 2\n0 4 2\n0 4 3\n0 5 3\n0 0"
    expected_output = "-1\n15\n12\n5955\n21431\n9298\n16392\n24774\n8840"
    run_pie_test_case("../p00449.py", input_content, expected_output)


def test_problem_p00449_1():
    input_content = "3 8\n1 3 1 10\n0 2 3\n1 2 3 20\n1 1 2 5\n0 3 2\n1 1 3 7\n1 2 1 9\n0 2 3\n5 16\n1 1 2 343750\n1 1 3 3343\n1 1 4 347392\n1 1 5 5497\n1 2 3 123394\n1 2 4 545492\n1 2 5 458\n1 3 4 343983\n1 3 5 843468\n1 4 5 15934\n0 2 1\n0 4 1\n0 3 2\n0 4 2\n0 4 3\n0 5 3\n0 0"
    expected_output = "-1\n15\n12\n5955\n21431\n9298\n16392\n24774\n8840"
    run_pie_test_case("../p00449.py", input_content, expected_output)
