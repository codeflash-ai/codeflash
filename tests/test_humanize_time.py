from codeflash.code_utils.time_utils import humanize_runtime


def test_humanize_runtime():
    assert humanize_runtime(0) == "0 nanoseconds"
    assert humanize_runtime(1000) == "1 microsecond"
    assert humanize_runtime(1000000) == "1 millisecond"
    assert humanize_runtime(1000000000) == "1 second"
    assert humanize_runtime(60000000000) == "1 minute"
    assert humanize_runtime(3600000000000) == "1 hour"

    assert humanize_runtime(1) == "1 nanoseconds"
    assert humanize_runtime(12) == "12 nanoseconds"
    assert humanize_runtime(123) == "123 nanoseconds"
    assert humanize_runtime(999) == "999 nanoseconds"
    assert humanize_runtime(1234) == "1.23 microsecond"
    assert humanize_runtime(12345) == "12.3 microseconds"
    assert humanize_runtime(123456) == "123 microseconds"
    assert humanize_runtime(1234567) == "1.23 millisecond"
    assert humanize_runtime(12345678) == "12.3 milliseconds"
    assert humanize_runtime(123456789) == "123 milliseconds"

    assert humanize_runtime(1234567891) == "1.23 second"
    assert humanize_runtime(12345678912) == "12.3 seconds"
    assert humanize_runtime(123456789123) == "2.06 minutes"
    assert humanize_runtime(1234567891234) == "20.6 minutes"
    assert humanize_runtime(12345678912345) == "3.43 hours"
