
import pytest

from video_anomaly_detection.cli import main


def test_main():
    with pytest.raises(SystemExit):
        try:
            main(['--help'])
        except SystemExit as e:
            assert e.code == 0
            raise
