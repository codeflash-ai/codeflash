from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02279_0():
    input_content = "13\n0 3 1 4 10\n1 2 2 3\n2 0\n3 0\n4 3 5 6 7\n5 0\n6 0\n7 2 8 9\n8 0\n9 0\n10 2 11 12\n11 0\n12 0"
    expected_output = "node 0: parent = -1, depth = 0, root, [1, 4, 10]\nnode 1: parent = 0, depth = 1, internal node, [2, 3]\nnode 2: parent = 1, depth = 2, leaf, []\nnode 3: parent = 1, depth = 2, leaf, []\nnode 4: parent = 0, depth = 1, internal node, [5, 6, 7]\nnode 5: parent = 4, depth = 2, leaf, []\nnode 6: parent = 4, depth = 2, leaf, []\nnode 7: parent = 4, depth = 2, internal node, [8, 9]\nnode 8: parent = 7, depth = 3, leaf, []\nnode 9: parent = 7, depth = 3, leaf, []\nnode 10: parent = 0, depth = 1, internal node, [11, 12]\nnode 11: parent = 10, depth = 2, leaf, []\nnode 12: parent = 10, depth = 2, leaf, []"
    run_pie_test_case("../p02279.py", input_content, expected_output)


def test_problem_p02279_1():
    input_content = "13\n0 3 1 4 10\n1 2 2 3\n2 0\n3 0\n4 3 5 6 7\n5 0\n6 0\n7 2 8 9\n8 0\n9 0\n10 2 11 12\n11 0\n12 0"
    expected_output = "node 0: parent = -1, depth = 0, root, [1, 4, 10]\nnode 1: parent = 0, depth = 1, internal node, [2, 3]\nnode 2: parent = 1, depth = 2, leaf, []\nnode 3: parent = 1, depth = 2, leaf, []\nnode 4: parent = 0, depth = 1, internal node, [5, 6, 7]\nnode 5: parent = 4, depth = 2, leaf, []\nnode 6: parent = 4, depth = 2, leaf, []\nnode 7: parent = 4, depth = 2, internal node, [8, 9]\nnode 8: parent = 7, depth = 3, leaf, []\nnode 9: parent = 7, depth = 3, leaf, []\nnode 10: parent = 0, depth = 1, internal node, [11, 12]\nnode 11: parent = 10, depth = 2, leaf, []\nnode 12: parent = 10, depth = 2, leaf, []"
    run_pie_test_case("../p02279.py", input_content, expected_output)


def test_problem_p02279_2():
    input_content = "4\n1 3 3 2 0\n0 0\n3 0\n2 0"
    expected_output = "node 0: parent = 1, depth = 1, leaf, []\nnode 1: parent = -1, depth = 0, root, [3, 2, 0]\nnode 2: parent = 1, depth = 1, leaf, []\nnode 3: parent = 1, depth = 1, leaf, []"
    run_pie_test_case("../p02279.py", input_content, expected_output)
