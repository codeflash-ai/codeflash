from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01224_0():
    input_content = "1\n2\n3\n4\n6\n12\n16\n28\n33550336\n99999998\n99999999\n100000000\n0"
    expected_output = "deficient number\ndeficient number\ndeficient number\ndeficient number\nperfect number\nabundant number\ndeficient number\nperfect number\nperfect number\ndeficient number\ndeficient number\nabundant number"
    run_pie_test_case("../p01224.py", input_content, expected_output)


def test_problem_p01224_1():
    input_content = "1\n2\n3\n4\n6\n12\n16\n28\n33550336\n99999998\n99999999\n100000000\n0"
    expected_output = "deficient number\ndeficient number\ndeficient number\ndeficient number\nperfect number\nabundant number\ndeficient number\nperfect number\nperfect number\ndeficient number\ndeficient number\nabundant number"
    run_pie_test_case("../p01224.py", input_content, expected_output)
