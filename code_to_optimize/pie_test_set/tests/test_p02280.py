from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02280_0():
    input_content = "9\n0 1 4\n1 2 3\n2 -1 -1\n3 -1 -1\n4 5 8\n5 6 7\n6 -1 -1\n7 -1 -1\n8 -1 -1"
    expected_output = "node 0: parent = -1, sibling = -1, degree = 2, depth = 0, height = 3, root\nnode 1: parent = 0, sibling = 4, degree = 2, depth = 1, height = 1, internal node\nnode 2: parent = 1, sibling = 3, degree = 0, depth = 2, height = 0, leaf\nnode 3: parent = 1, sibling = 2, degree = 0, depth = 2, height = 0, leaf\nnode 4: parent = 0, sibling = 1, degree = 2, depth = 1, height = 2, internal node\nnode 5: parent = 4, sibling = 8, degree = 2, depth = 2, height = 1, internal node\nnode 6: parent = 5, sibling = 7, degree = 0, depth = 3, height = 0, leaf\nnode 7: parent = 5, sibling = 6, degree = 0, depth = 3, height = 0, leaf\nnode 8: parent = 4, sibling = 5, degree = 0, depth = 2, height = 0, leaf"
    run_pie_test_case("../p02280.py", input_content, expected_output)


def test_problem_p02280_1():
    input_content = "9\n0 1 4\n1 2 3\n2 -1 -1\n3 -1 -1\n4 5 8\n5 6 7\n6 -1 -1\n7 -1 -1\n8 -1 -1"
    expected_output = "node 0: parent = -1, sibling = -1, degree = 2, depth = 0, height = 3, root\nnode 1: parent = 0, sibling = 4, degree = 2, depth = 1, height = 1, internal node\nnode 2: parent = 1, sibling = 3, degree = 0, depth = 2, height = 0, leaf\nnode 3: parent = 1, sibling = 2, degree = 0, depth = 2, height = 0, leaf\nnode 4: parent = 0, sibling = 1, degree = 2, depth = 1, height = 2, internal node\nnode 5: parent = 4, sibling = 8, degree = 2, depth = 2, height = 1, internal node\nnode 6: parent = 5, sibling = 7, degree = 0, depth = 3, height = 0, leaf\nnode 7: parent = 5, sibling = 6, degree = 0, depth = 3, height = 0, leaf\nnode 8: parent = 4, sibling = 5, degree = 0, depth = 2, height = 0, leaf"
    run_pie_test_case("../p02280.py", input_content, expected_output)
