import time


def sleepfunc_sequence(t) -> int:
    total_sleep_time = 0.0
    for i in range(t + 1):  # Loop from 0 to n inclusive
        sleep_duration = i / 100
        time.sleep(sleep_duration)
        total_sleep_time += sleep_duration
    return 1
