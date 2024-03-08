from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02396_0():
    input_content = "3\n5\n11\n7\n8\n19\n0"
    expected_output = "Case 1: 3\nCase 2: 5\nCase 3: 11\nCase 4: 7\nCase 5: 8\nCase 6: 19"
    run_pie_test_case("../p02396.py", input_content, expected_output)


def test_problem_p02396_1():
    input_content = "3\n5\n11\n7\n8\n19\n0"
    expected_output = "Case 1: 3\nCase 2: 5\nCase 3: 11\nCase 4: 7\nCase 5: 8\nCase 6: 19"
    run_pie_test_case("../p02396.py", input_content, expected_output)
