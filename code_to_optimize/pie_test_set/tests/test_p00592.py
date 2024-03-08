from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00592_0():
    input_content = "1 2100 2400\n1\n2130 2200\n3 2100 2400\n3\n2100 2130 2200 2230 2300 2330\n2\n2130 2200 2330 2400\n2\n2100 2130 2330 2400\n0 0 0"
    expected_output = "120\n180"
    run_pie_test_case("../p00592.py", input_content, expected_output)
