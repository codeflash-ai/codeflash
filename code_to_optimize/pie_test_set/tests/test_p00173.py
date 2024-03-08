from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00173_0():
    input_content = "1a 132 243\n1c 324 183\n1f 93 199\n2b 372 163\n2c 229 293\n2e 391 206\n3a 118 168\n3b 263 293\n3d 281 102"
    expected_output = "1a 375 99300\n1c 507 119700\n1f 292 78300\n2b 535 123300\n2c 522 133700\n2e 597 140000\n3a 286 74000\n3b 556 140500\n3d 383 86800"
    run_pie_test_case("../p00173.py", input_content, expected_output)


def test_problem_p00173_1():
    input_content = "1a 132 243\n1c 324 183\n1f 93 199\n2b 372 163\n2c 229 293\n2e 391 206\n3a 118 168\n3b 263 293\n3d 281 102"
    expected_output = "1a 375 99300\n1c 507 119700\n1f 292 78300\n2b 535 123300\n2c 522 133700\n2e 597 140000\n3a 286 74000\n3b 556 140500\n3d 383 86800"
    run_pie_test_case("../p00173.py", input_content, expected_output)
