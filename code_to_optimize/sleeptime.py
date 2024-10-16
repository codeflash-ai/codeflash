import time

from code_to_optimize.use_cosine_similarity_from_other_file import CACHED_TESTS


def sleepfunc(t):
    time.sleep(t / 100)
    return 1
