# tests/test_event_bus.py

import importlib.util
import pathlib

import pytest

module_path = pathlib.Path(__file__).resolve().parents[1] / 'event_bus.py'
spec = importlib.util.spec_from_file_location('workspace_event_bus', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EventBus = module.EventBus

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
