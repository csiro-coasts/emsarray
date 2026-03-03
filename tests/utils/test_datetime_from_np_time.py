import datetime

import freezegun
import numpy
import pytest

from emsarray import utils


@pytest.mark.parametrize('tz_offset', [0, 10, -4])
def test_datetime_from_np_time(tz_offset: int):
    # Change the system timezone to `tz_offset`
    with freezegun.freeze_time(tz_offset=tz_offset):
        np_time = numpy.datetime64('2025-08-18T12:05:00.123456')

        # Test that converting works using the UTC default timezone,
        # regardless of system timezone
        py_time_utc = utils.datetime_from_np_time(np_time)
        assert py_time_utc == datetime.datetime(2025, 8, 18, 12, 5, 0, 123456, tzinfo=datetime.UTC)

        # Test that converting works when interpreted in the system timezone.
        py_tz_system = datetime.timezone(datetime.timedelta(hours=tz_offset))
        py_time_local = utils.datetime_from_np_time(np_time, tz=py_tz_system)
        assert py_time_local == datetime.datetime(2025, 8, 18, 12, 5, 0, 123456, tzinfo=py_tz_system)

        # Test that converting works when using some other arbitrary timezone.
        py_tz_eucla = datetime.timezone(datetime.timedelta(hours=8, minutes=45))
        py_time_eucla = utils.datetime_from_np_time(np_time, tz=py_tz_eucla)
        assert py_time_eucla == datetime.datetime(2025, 8, 18, 12, 5, 0, 123456, tzinfo=py_tz_eucla)
