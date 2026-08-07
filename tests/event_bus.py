import importlib.util
import pathlib
import unittest

module_path = pathlib.Path(__file__).resolve().parents[1] / 'event_bus.py'
spec = importlib.util.spec_from_file_location('workspace_event_bus', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EventBus = module.EventBus


class TestEventBus(unittest.TestCase):

    def test_event_bus_init(self):
        bus = EventBus()
        self.assertIsNotNone(bus)

    def test_event_bus_publish(self):
        bus = EventBus()
        received = []

        def callback(event):
            received.append(event)

        bus.subscribe(callback)
        bus.publish('event')
        self.assertEqual(received, ['event'])

    def test_event_bus_subscribe(self):
        bus = EventBus()

        def callback(event):
            pass

        bus.subscribe(callback)
        self.assertEqual(bus.subscribers, [callback])

    def test_event_bus_unsubscribe(self):
        bus = EventBus()

        def callback(event):
            pass

        bus.subscribe(callback)
        bus.unsubscribe(callback)
        self.assertNotIn(callback, bus.subscribers)
