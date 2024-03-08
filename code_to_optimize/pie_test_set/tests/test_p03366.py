from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03366_0():
    input_content = "3 2\n1 3\n3 4\n4 2"
    expected_output = "4"
    run_pie_test_case("../p03366.py", input_content, expected_output)


def test_problem_p03366_1():
    input_content = "15 409904902\n94198000 15017\n117995501 7656764\n275583856 313263626\n284496300 356635175\n324233841 607\n360631781 148\n472103717 5224\n497641071 34695\n522945827 816241\n554305668 32\n623788284 22832\n667409501 124410641\n876731548 12078\n904557302 291749534\n918215789 5"
    expected_output = "2397671583"
    run_pie_test_case("../p03366.py", input_content, expected_output)


def test_problem_p03366_2():
    input_content = "3 2\n1 3\n3 4\n4 2"
    expected_output = "4"
    run_pie_test_case("../p03366.py", input_content, expected_output)


def test_problem_p03366_3():
    input_content = "6 4\n1 10\n2 1000\n3 100000\n5 1000000\n6 10000\n7 100"
    expected_output = "21"
    run_pie_test_case("../p03366.py", input_content, expected_output)
