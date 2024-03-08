from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00164_0():
    input_content = "4\n3 1 4 2\n3\n4 3 2\n0"
    expected_output = "31\n28\n26\n25\n21\n17\n16\n14\n11\n8\n6\n5\n1\n0\n31\n27\n26\n23\n21\n19\n16\n12\n11\n8\n6\n4\n1\n0"
    run_pie_test_case("../p00164.py", input_content, expected_output)
