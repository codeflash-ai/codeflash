from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01753_0():
    input_content = (
        "5 1\n0 10 0 5 2\n0 20 0 5 12\n0 30 0 5 22\n0 40 0 5 32\n0 50 0 5 42\n0 0 0 0 60 0"
    )
    expected_output = "110"
    run_pie_test_case("../p01753.py", input_content, expected_output)


def test_problem_p01753_1():
    input_content = "5 5\n-38 -71 -293 75 1\n-158 -38 -405 66 1\n-236 -303 157 266 1\n316 26 411 190 1\n207 -312 -27 196 1\n-50 292 -375 -401 389 -389\n460 278 409 -329 -303 411\n215 -220 -200 309 -474 300\n261 -494 -87 -300 123 -463\n386 378 486 -443 -64 299"
    expected_output = "0\n2\n1\n3\n0"
    run_pie_test_case("../p01753.py", input_content, expected_output)


def test_problem_p01753_2():
    input_content = (
        "5 1\n0 10 0 5 2\n0 20 0 5 12\n0 30 0 5 22\n0 40 0 5 32\n0 50 0 5 42\n0 0 0 0 60 0"
    )
    expected_output = "110"
    run_pie_test_case("../p01753.py", input_content, expected_output)


def test_problem_p01753_3():
    input_content = "1 1\n10 5 0 5 9\n0 0 0 9 12 0"
    expected_output = "9"
    run_pie_test_case("../p01753.py", input_content, expected_output)
