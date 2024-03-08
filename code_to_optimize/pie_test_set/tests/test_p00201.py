from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00201_0():
    input_content = "8\nwood 3000\nstring 800\nrice 36\nwater 0\nracket 5000\nmicrophone 9800\nonigiri 140\nguitar 98000\n3\nracket 2 wood string\nonigiri 2 rice water\nguitar 3 racket microphone onigiri\nguitar\n1\ncomputer 300000\n0\ncomputer\n0"
    expected_output = "13636\n300000"
    run_pie_test_case("../p00201.py", input_content, expected_output)


def test_problem_p00201_1():
    input_content = "8\nwood 3000\nstring 800\nrice 36\nwater 0\nracket 5000\nmicrophone 9800\nonigiri 140\nguitar 98000\n3\nracket 2 wood string\nonigiri 2 rice water\nguitar 3 racket microphone onigiri\nguitar\n1\ncomputer 300000\n0\ncomputer\n0"
    expected_output = "13636\n300000"
    run_pie_test_case("../p00201.py", input_content, expected_output)
