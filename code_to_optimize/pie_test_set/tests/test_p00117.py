from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00117_0():
    input_content = (
        "6\n8\n1,2,2,2\n1,3,4,3\n1,4,4,2\n2,5,3,2\n3,4,4,2\n3,6,1,2\n4,6,1,1\n5,6,1,2\n2,4,50,30"
    )
    expected_output = "11"
    run_pie_test_case("../p00117.py", input_content, expected_output)


def test_problem_p00117_1():
    input_content = (
        "6\n8\n1,2,2,2\n1,3,4,3\n1,4,4,2\n2,5,3,2\n3,4,4,2\n3,6,1,2\n4,6,1,1\n5,6,1,2\n2,4,50,30"
    )
    expected_output = "11"
    run_pie_test_case("../p00117.py", input_content, expected_output)
