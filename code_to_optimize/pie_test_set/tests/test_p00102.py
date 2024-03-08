from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00102_0():
    input_content = "4\n52 96 15 20\n86 22 35 45\n45 78 54 36\n16 86 74 55\n4\n52 96 15 20\n86 22 35 45\n45 78 54 36\n16 86 74 55\n0"
    expected_output = "52   96   15   20  183\n   86   22   35   45  188\n   45   78   54   36  213\n   16   86   74   55  231\n  199  282  178  156  815\n   52   96   15   20  183\n   86   22   35   45  188\n   45   78   54   36  213\n   16   86   74   55  231\n  199  282  178  156  815"
    run_pie_test_case("../p00102.py", input_content, expected_output)


def test_problem_p00102_1():
    input_content = "4\n52 96 15 20\n86 22 35 45\n45 78 54 36\n16 86 74 55\n4\n52 96 15 20\n86 22 35 45\n45 78 54 36\n16 86 74 55\n0"
    expected_output = "52   96   15   20  183\n   86   22   35   45  188\n   45   78   54   36  213\n   16   86   74   55  231\n  199  282  178  156  815\n   52   96   15   20  183\n   86   22   35   45  188\n   45   78   54   36  213\n   16   86   74   55  231\n  199  282  178  156  815"
    run_pie_test_case("../p00102.py", input_content, expected_output)
