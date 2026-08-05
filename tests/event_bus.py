import unittest
from unittest.mock import MagicMock, patch
from event_bus import EventBus

class TestEventBus(unittest.TestCase):

    def test_event_bus_init(self):
        bus = EventBus()
        self.assertIsNotNone(bus)

    @patch('event_bus.Event')
    def test_event_bus_publish(self, mock_Event):
        bus = EventBus()
        event = mock_Event.return_value
        bus.publish(event)
        event.assert_called_once()

    @patch('event_bus.Event')
    def test_event_bus_subscribe(self, mock_Event):
        bus = EventBus()
        def callback(event):
            pass
        bus.subscribe(callback)
        self.assertEqual(bus.subscribers, [callback])

    @patch('event_bus.Event')
    def test_event_bus_unsubscribe(self, mock_Event):
        bus = EventBus()
        def callback(event):
            pass
        bus.subscribe(callback)
        bus.unsubscribe(callback)
        self.assertNotIn(callback, bus.subscribers)
