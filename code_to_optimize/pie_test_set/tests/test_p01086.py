from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01086_0():
    input_content = "9\ndo\nthe\nbest\nand\nenjoy\ntoday\nat\nacm\nicpc\n14\noh\nyes\nby\nfar\nit\nis\nwow\nso\nbad\nto\nme\nyou\nknow\nhey\n15\nabcde\nfghijkl\nmnopq\nrstuvwx\nyzz\nabcde\nfghijkl\nmnopq\nrstuvwx\nyz\nabcde\nfghijkl\nmnopq\nrstuvwx\nyz\n0"
    expected_output = "1\n2\n6"
    run_pie_test_case("../p01086.py", input_content, expected_output)


def test_problem_p01086_1():
    input_content = "9\ndo\nthe\nbest\nand\nenjoy\ntoday\nat\nacm\nicpc\n14\noh\nyes\nby\nfar\nit\nis\nwow\nso\nbad\nto\nme\nyou\nknow\nhey\n15\nabcde\nfghijkl\nmnopq\nrstuvwx\nyzz\nabcde\nfghijkl\nmnopq\nrstuvwx\nyz\nabcde\nfghijkl\nmnopq\nrstuvwx\nyz\n0"
    expected_output = "1\n2\n6"
    run_pie_test_case("../p01086.py", input_content, expected_output)
