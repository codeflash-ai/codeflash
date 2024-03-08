from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00161_0():
    input_content = "8\n34001 3 20 3 8 6 27 2 25\n20941 3 5 2 41 7 19 2 42\n90585 4 8 3 12 6 46 2 34\n92201 3 28 2 47 6 37 2 58\n10001 3 50 2 42 7 12 2 54\n63812 4 11 3 11 6 53 2 22\n54092 3 33 2 54 6 18 2 19\n25012 3 44 2 58 6 45 2 46\n4\n1 3 23 1 23 1 34 4 44\n2 5 12 2 12 3 41 2 29\n3 5 24 1 24 2 0 3 35\n4 4 49 2 22 4 41 4 23\n0"
    expected_output = "54092\n34001\n10001\n1\n3\n2"
    run_pie_test_case("../p00161.py", input_content, expected_output)


def test_problem_p00161_1():
    input_content = "8\n34001 3 20 3 8 6 27 2 25\n20941 3 5 2 41 7 19 2 42\n90585 4 8 3 12 6 46 2 34\n92201 3 28 2 47 6 37 2 58\n10001 3 50 2 42 7 12 2 54\n63812 4 11 3 11 6 53 2 22\n54092 3 33 2 54 6 18 2 19\n25012 3 44 2 58 6 45 2 46\n4\n1 3 23 1 23 1 34 4 44\n2 5 12 2 12 3 41 2 29\n3 5 24 1 24 2 0 3 35\n4 4 49 2 22 4 41 4 23\n0"
    expected_output = "54092\n34001\n10001\n1\n3\n2"
    run_pie_test_case("../p00161.py", input_content, expected_output)
