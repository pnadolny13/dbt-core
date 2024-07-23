import os

from dbt.cli.main import dbtRunner
from dbt.events.types import ResourceReport
from dbt_common.events.base_types import EventLevel
from tests.functional.events.event_monitor import EventMonitor, ExpectedEvent


def test_performance_report(project):

    monitor = EventMonitor(
        [
            ExpectedEvent(ResourceReport, info={"level": EventLevel.DEBUG}, data={}),
        ]
    )

    runner = dbtRunner(callbacks=[monitor])
    runner.invoke(["run"])

    assert monitor.all_events_matched()

    try:
        os.environ["DBT_SHOW_RESOURCE_REPORT"] = "1"

        # With the appropriate env var set, ResourceReport should be info level.
        # This allows this fairly technical log line to be omitted by default
        # but still available in production scenarios.
        monitor = EventMonitor(
            [
                ExpectedEvent(ResourceReport, info={"level": EventLevel.INFO}, data={}),
            ]
        )

        runner = dbtRunner(callbacks=[monitor])
        runner.invoke(["run"])

        assert monitor.all_events_matched()
    finally:
        del os.environ["DBT_SHOW_RESOURCE_REPORT"]
