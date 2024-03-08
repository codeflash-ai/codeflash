from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02289_0():
    input_content = (
        "insert 8\ninsert 2\nextract\ninsert 10\nextract\ninsert 11\nextract\nextract\nend"
    )
    expected_output = "8\n10\n11\n2"
    run_pie_test_case("../p02289.py", input_content, expected_output)


def test_problem_p02289_1():
    input_content = (
        "insert 8\ninsert 2\nextract\ninsert 10\nextract\ninsert 11\nextract\nextract\nend"
    )
    expected_output = "8\n10\n11\n2"
    run_pie_test_case("../p02289.py", input_content, expected_output)
