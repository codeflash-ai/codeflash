# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://www.mozilla.org/en-US/MPL/2.0/.
from __future__ import annotations

# System Imports
import logging
import re
import shutil
import time
import warnings
from typing import TYPE_CHECKING
from unittest import TestCase

# PyTest Imports
import pytest
from pluggy import HookspecMarker

if TYPE_CHECKING:
    from _pytest.config import Config, Parser
    from _pytest.main import Session
    from _pytest.python import Metafunc

SECONDS_IN_HOUR: float = 3600
SECONDS_IN_MINUTE: float = 60
SHORTEST_AMOUNT_OF_TIME: float = 0
hookspec = HookspecMarker("pytest")


class InvalidTimeParameterError(Exception):
    pass


class UnexpectedError(Exception):
    pass


def pytest_addoption(parser: Parser) -> None:
    """Add our command line options."""
    pytest_loop = parser.getgroup("loop")
    pytest_loop.addoption(
        "--delay",
        action="store",
        default=0,
        type=float,
        help="The amount of time to wait between each test loop.",
    )
    pytest_loop.addoption(
        "--hours",
        action="store",
        default=0,
        type=float,
        help="The number of hours to loop the tests for.",
    )
    pytest_loop.addoption(
        "--minutes",
        action="store",
        default=0,
        type=float,
        help="The number of minutes to loop the tests for.",
    )
    pytest_loop.addoption(
        "--seconds",
        action="store",
        default=0,
        type=float,
        help="The number of seconds to loop the tests for.",
    )

    pytest_loop.addoption(
        "--loop",
        action="store",
        default=1,
        type=int,
        help="The number of times to loop each test",
    )

    pytest_loop.addoption(
        "--loop-scope",
        action="store",
        default="function",
        type=str,
        choices=("function", "class", "module", "session"),
        help="Scope for looping tests",
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "loop(n): run the given test function `n` times.",
    )
    config.pluginmanager.register(PyTest_Loop(config), PyTest_Loop.name)


class PyTest_Loop:
    name = "pytest-loop"

    def __init__(self, config: Config) -> None:
        # turn debug prints on only if "-vv" or more passed
        level = logging.DEBUG if config.option.verbose > 1 else logging.INFO
        logging.basicConfig(level=level)
        self.logger = logging.getLogger(self.name)

    @hookspec(firstresult=True)
    def pytest_runtestloop(self, session: Session) -> bool:
        """Reimplement the test loop but loop for the user defined amount of time."""
        if (
            session.testsfailed
            and not session.config.option.continue_on_collection_errors
        ):
            raise session.Interrupted(
                "%d error%s during collection"
                % (session.testsfailed, "s" if session.testsfailed != 1 else ""),
            )

        if session.config.option.collectonly:
            return True

        start_time: float = time.time()
        total_time: float = self._get_total_time(session)

        count: int = 0

        while (
            total_time >= SHORTEST_AMOUNT_OF_TIME
        ):  # need to run at least one for normal tests
            count += 1
            total_time = self._get_total_time(session)

            for index, item in enumerate(session.items):
                item: pytest.Item = item
                item._report_sections.clear()  # clear reports for new test

                if total_time > SHORTEST_AMOUNT_OF_TIME:
                    self._print_loop_count(count)
                    item._nodeid = self._set_nodeid(item._nodeid, count)

                next_item: pytest.Item = (
                    session.items[index + 1] if index + 1 < len(session.items) else None
                )
                item.config.hook.pytest_runtest_protocol(item=item, nextitem=next_item)
                if session.shouldfail:
                    raise session.Failed(session.shouldfail)
                if session.shouldstop:
                    raise session.Interrupted(session.shouldstop)
            if self._timed_out(session, start_time):
                break  # exit loop
            time.sleep(self._get_delay_time(session))
        return True

    def _set_nodeid(self, nodeid: str, count: int) -> str:
        """Helper function to set the loop count when using duration.

        :param nodeid: Name of test function.
        :param count: Current loop count.
        :return: Formatted string for test name.
        """
        pattern = r"\[ \d+ \]"
        run_str = f"[ {count} ]"
        if re.search(pattern, nodeid):
            nodeid = re.sub(pattern, run_str, nodeid)
        else:
            nodeid = nodeid + run_str
        return nodeid

    def _get_delay_time(self, session: Session) -> float:
        """Helper function to extract the delay time from the session.

        :param session: Pytest session object.
        :return: Returns the delay time for each test loop.
        """
        return session.config.option.delay

    def _get_total_time(self, session: Session) -> float:
        """Takes all the user available time options, adds them and returns it in seconds.

        :param session: Pytest session object.
        :return: Returns total amount of time in seconds.
        """
        hours_in_seconds = session.config.option.hours * SECONDS_IN_HOUR
        minutes_in_seconds = session.config.option.minutes * SECONDS_IN_MINUTE
        seconds = session.config.option.seconds
        total_time = hours_in_seconds + minutes_in_seconds + seconds
        if total_time < SHORTEST_AMOUNT_OF_TIME:
            raise InvalidTimeParameterError(
                f"Total time cannot be less than: {SHORTEST_AMOUNT_OF_TIME}!",
            )
        return total_time

    def _timed_out(self, session: Session, start_time: float) -> bool:
        """Helper function to check if the user specified amount of time has lapsed.

        :param session: Pytest session object.
        :return: Returns True if the timeout has expired, False otherwise.
        """
        return time.time() - start_time > self._get_total_time(session)

    def _print_loop_count(self, count: int) -> None:
        """Helper function to simply print out what loop number we're on.

        :param count: The number to print.
        :return: None.
        """
        column_length = shutil.get_terminal_size().columns
        print("\n")
        print(f" Loop # {count} ".center(column_length, "="))
        print("\n")

    @pytest.fixture
    def __pytest_loop_step_number(self, request: pytest.FixtureRequest) -> int:
        """Fixture function to set step number for loop.

        :param request: The number to print.
        :return: request.param.
        """
        marker = request.node.get_closest_marker("loop")
        count = marker and marker.args[0] or request.config.option.loop
        if count > 1:
            try:
                return request.param
            except AttributeError:
                if issubclass(request.cls, TestCase):
                    warnings.warn("Repeating unittest class tests not supported")
                else:
                    raise UnexpectedError(
                        "This call couldn't work with pytest-loop. "
                        "Please consider raising an issue with your usage.",
                    )
        return count

    @pytest.hookimpl(trylast=True)
    def pytest_generate_tests(self, metafunc: Metafunc) -> None:
        """Hook function to create tests based on loop value.

        :param metafunc: pytest metafunction
        :return: None.
        """
        count = metafunc.config.option.loop
        m = metafunc.definition.get_closest_marker("loop")

        if m is not None:
            count = int(m.args[0])
        if count > 1:
            metafunc.fixturenames.append("__pytest_loop_step_number")

            def make_progress_id(i: int, n: int = count) -> str:
                return f"{n}/{i+1}"

            scope = metafunc.config.option.loop_scope
            metafunc.parametrize(
                "__pytest_loop_step_number",
                range(count),
                indirect=True,
                ids=make_progress_id,
                scope=scope,
            )
