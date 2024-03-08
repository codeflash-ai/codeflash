from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00179_0():
    input_content = "rbgrg\nrbbgbbr\nbgr\nbgrbrgbr\nbggrgbgrr\ngbrggrbggr\nrrrrr\nbgbr\n0"
    expected_output = "5\n7\n1\n6\nNA\n8\n0\n4"
    run_pie_test_case("../p00179.py", input_content, expected_output)


def test_problem_p00179_1():
    input_content = "rbgrg\nrbbgbbr\nbgr\nbgrbrgbr\nbggrgbgrr\ngbrggrbggr\nrrrrr\nbgbr\n0"
    expected_output = "5\n7\n1\n6\nNA\n8\n0\n4"
    run_pie_test_case("../p00179.py", input_content, expected_output)
