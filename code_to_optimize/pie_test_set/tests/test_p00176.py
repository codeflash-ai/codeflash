from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00176_0():
    input_content = "#ffe085\n#787878\n#decade\n#ff55ff\n0"
    expected_output = "white\nblack\nwhite\nfuchsia"
    run_pie_test_case("../p00176.py", input_content, expected_output)


def test_problem_p00176_1():
    input_content = "#ffe085\n#787878\n#decade\n#ff55ff\n0"
    expected_output = "white\nblack\nwhite\nfuchsia"
    run_pie_test_case("../p00176.py", input_content, expected_output)


def test_problem_p00176_2():
    input_content = "ffe085\n787878\ndecade\nff55ff\n0"
    expected_output = "white\nblack\nwhite\nfuchsia"
    run_pie_test_case("../p00176.py", input_content, expected_output)
