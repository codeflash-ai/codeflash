from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02244_0():
    input_content = "2\n2 2\n5 3"
    expected_output = (
        "......Q.\nQ.......\n..Q.....\n.......Q\n.....Q..\n...Q....\n.Q......\n....Q..."
    )
    run_pie_test_case("../p02244.py", input_content, expected_output)


def test_problem_p02244_1():
    input_content = "2\n2 2\n5 3"
    expected_output = (
        "......Q.\nQ.......\n..Q.....\n.......Q\n.....Q..\n...Q....\n.Q......\n....Q..."
    )
    run_pie_test_case("../p02244.py", input_content, expected_output)
