from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02281_0():
    input_content = "9\n0 1 4\n1 2 3\n2 -1 -1\n3 -1 -1\n4 5 8\n5 6 7\n6 -1 -1\n7 -1 -1\n8 -1 -1"
    expected_output = (
        "Preorder\n 0 1 2 3 4 5 6 7 8\nInorder\n 2 1 3 0 6 5 7 4 8\nPostorder\n 2 3 1 6 7 5 8 4 0"
    )
    run_pie_test_case("../p02281.py", input_content, expected_output)


def test_problem_p02281_1():
    input_content = "9\n0 1 4\n1 2 3\n2 -1 -1\n3 -1 -1\n4 5 8\n5 6 7\n6 -1 -1\n7 -1 -1\n8 -1 -1"
    expected_output = (
        "Preorder\n 0 1 2 3 4 5 6 7 8\nInorder\n 2 1 3 0 6 5 7 4 8\nPostorder\n 2 3 1 6 7 5 8 4 0"
    )
    run_pie_test_case("../p02281.py", input_content, expected_output)
