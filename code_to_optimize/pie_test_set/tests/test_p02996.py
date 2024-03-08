from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02996_0():
    input_content = "5\n2 4\n1 9\n1 8\n4 9\n3 12"
    expected_output = "Yes"
    run_pie_test_case("../p02996.py", input_content, expected_output)


def test_problem_p02996_1():
    input_content = "3\n334 1000\n334 1000\n334 1000"
    expected_output = "No"
    run_pie_test_case("../p02996.py", input_content, expected_output)


def test_problem_p02996_2():
    input_content = "5\n2 4\n1 9\n1 8\n4 9\n3 12"
    expected_output = "Yes"
    run_pie_test_case("../p02996.py", input_content, expected_output)


def test_problem_p02996_3():
    input_content = "30\n384 8895\n1725 9791\n170 1024\n4 11105\n2 6\n578 1815\n702 3352\n143 5141\n1420 6980\n24 1602\n849 999\n76 7586\n85 5570\n444 4991\n719 11090\n470 10708\n1137 4547\n455 9003\n110 9901\n15 8578\n368 3692\n104 1286\n3 4\n366 12143\n7 6649\n610 2374\n152 7324\n4 7042\n292 11386\n334 5720"
    expected_output = "Yes"
    run_pie_test_case("../p02996.py", input_content, expected_output)
