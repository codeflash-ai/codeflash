from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02905_0():
    input_content = "3\n2 4 6"
    expected_output = "22"
    run_pie_test_case("../p02905.py", input_content, expected_output)


def test_problem_p02905_1():
    input_content = "3\n2 4 6"
    expected_output = "22"
    run_pie_test_case("../p02905.py", input_content, expected_output)


def test_problem_p02905_2():
    input_content = "8\n1 2 3 4 6 8 12 12"
    expected_output = "313"
    run_pie_test_case("../p02905.py", input_content, expected_output)


def test_problem_p02905_3():
    input_content = "10\n356822 296174 484500 710640 518322 888250 259161 609120 592348 713644"
    expected_output = "353891724"
    run_pie_test_case("../p02905.py", input_content, expected_output)
