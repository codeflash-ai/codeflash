from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02269_0():
    input_content = "5\ninsert A\ninsert T\ninsert C\nfind G\nfind A"
    expected_output = "no\nyes"
    run_pie_test_case("../p02269.py", input_content, expected_output)


def test_problem_p02269_1():
    input_content = "5\ninsert A\ninsert T\ninsert C\nfind G\nfind A"
    expected_output = "no\nyes"
    run_pie_test_case("../p02269.py", input_content, expected_output)


def test_problem_p02269_2():
    input_content = "13\ninsert AAA\ninsert AAC\ninsert AGA\ninsert AGG\ninsert TTT\nfind AAA\nfind CCC\nfind CCC\ninsert CCC\nfind CCC\ninsert T\nfind TTT\nfind T"
    expected_output = "yes\nno\nno\nyes\nyes\nyes"
    run_pie_test_case("../p02269.py", input_content, expected_output)
