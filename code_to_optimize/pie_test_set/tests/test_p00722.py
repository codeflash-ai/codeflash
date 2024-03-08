from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00722_0():
    input_content = "367 186 151\n179 10 203\n271 37 39\n103 230 1\n27 104 185\n253 50 85\n1 1 1\n9075 337 210\n307 24 79\n331 221 177\n259 170 40\n269 58 102\n0 0 0"
    expected_output = "92809\n6709\n12037\n103\n93523\n14503\n2\n899429\n5107\n412717\n22699\n25673"
    run_pie_test_case("../p00722.py", input_content, expected_output)


def test_problem_p00722_1():
    input_content = "367 186 151\n179 10 203\n271 37 39\n103 230 1\n27 104 185\n253 50 85\n1 1 1\n9075 337 210\n307 24 79\n331 221 177\n259 170 40\n269 58 102\n0 0 0"
    expected_output = "92809\n6709\n12037\n103\n93523\n14503\n2\n899429\n5107\n412717\n22699\n25673"
    run_pie_test_case("../p00722.py", input_content, expected_output)
