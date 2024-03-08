from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01138_0():
    input_content = "3\n05:47:15 09:54:40\n12:12:59 12:13:00\n16:30:20 21:18:53\n6\n00:00:00 03:00:00\n01:00:00 03:00:00\n02:00:00 03:00:00\n03:00:00 04:00:00\n03:00:00 05:00:00\n03:00:00 06:00:00\n0"
    expected_output = "1\n3"
    run_pie_test_case("../p01138.py", input_content, expected_output)


def test_problem_p01138_1():
    input_content = "3\n05:47:15 09:54:40\n12:12:59 12:13:00\n16:30:20 21:18:53\n6\n00:00:00 03:00:00\n01:00:00 03:00:00\n02:00:00 03:00:00\n03:00:00 04:00:00\n03:00:00 05:00:00\n03:00:00 06:00:00\n0"
    expected_output = "1\n3"
    run_pie_test_case("../p01138.py", input_content, expected_output)
