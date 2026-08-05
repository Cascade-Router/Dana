# tests/test_event_bus.py

import pytest
from event_bus import EventBus

def test_event_bus_created():
    event_bus = EventBus()
    assert hasattr(event_bus, 'subscribe')
    assert hasattr(event_bus, 'unsubscribe')

def test_event_bus_subscribe_unsubscribe():
    event_bus = EventBus()
    def callback(event):
        pass
    event_bus.subscribe('test_event', callback)
    event_bus.unsubscribe('test_event', callback)
    with pytest.raises(ValueError):
        event_bus.unsubscribe('test_event', callback)

def test_event_bus_publish():
    event_bus = EventBus()
    def callback(event):
        assert event == 'test_event'
    event_bus.subscribe('test_event', callback)
    event_bus.publish('test_event')
