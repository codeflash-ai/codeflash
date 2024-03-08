from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02283_0():
    input_content = (
        "8\ninsert 30\ninsert 88\ninsert 12\ninsert 1\ninsert 20\ninsert 17\ninsert 25\nprint"
    )
    expected_output = "1 12 17 20 25 30 88\n 30 12 1 20 17 25 88"
    run_pie_test_case("../p02283.py", input_content, expected_output)


def test_problem_p02283_1():
    input_content = (
        "8\ninsert 30\ninsert 88\ninsert 12\ninsert 1\ninsert 20\ninsert 17\ninsert 25\nprint"
    )
    expected_output = "1 12 17 20 25 30 88\n 30 12 1 20 17 25 88"
    run_pie_test_case("../p02283.py", input_content, expected_output)
