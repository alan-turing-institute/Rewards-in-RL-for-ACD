import os.path
from logging import getLogger
from uuid import uuid4

import pytest

from yawning_titan import LOG_FILE_PATH


@pytest.mark.unit_test
def test_log_file_location():
    """
    Tests the application log_initial file.

    Tests the Yawning-Titan logging config by asserting that the log_initial file exists in the assumed place, and by testing
    that a unique entry in the log_initial is written by reading it back in, parsing the log_initial, and comparing the message.
    """
    expected_location = LOG_FILE_PATH
    _LOGGER = getLogger(__name__)

    assert os.path.isfile(expected_location)

    expected_msg = f"test unique message {uuid4()}"
    _LOGGER.info(expected_msg)

    with open(expected_location, "r") as log_file:
        logs = log_file.read().splitlines()
        last_log = logs[-1]

    assert expected_msg == last_log.split("::")[-1]
