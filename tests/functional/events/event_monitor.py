import dataclasses
from typing import Any, List, Type

from dbt_common.events.base_types import BaseEvent, EventMsg


@dataclasses.dataclass
class ExpectedEvent:
    """Describes an expected event. An event matches this expectation if it has
    the correct type and the properties specified in the info and data attributes
    are also present in the event fired at runtime, with the exact same values.
    Only the specified properties are checked for equality. If the runtime event
    has other properties, they are ignored."""

    event_type: Type[BaseEvent]
    info: dict[str, Any]
    data: dict[str, Any]

    def matches(self, event: EventMsg) -> bool:
        if self.event_type.__name__ != event.info.name:
            return False

        try:
            for k, v in self.info.items():
                actual_value = getattr(event.info, k)
                if actual_value != v:
                    return False

            for k, v in self.data.items():
                actual_value = getattr(event.data, k)
                if actual_value != v:
                    return False
        except Exception:
            return False

        return True


class EventMonitor:
    """This class monitors dbt during an invocation to ensure that a list of
    expected events are fired."""

    def __init__(self, expected_events: List[ExpectedEvent]) -> None:
        self.expected_events = expected_events

    def __call__(self, event: EventMsg) -> None:
        for expected in self.expected_events:
            if expected.matches(event):
                self.expected_events.remove(expected)
                break

    def all_events_matched(self) -> bool:
        return len(self.expected_events) == 0
