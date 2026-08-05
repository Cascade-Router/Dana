class Bus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type, subscriber):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(subscriber)

    def unsubscribe(self, event_type, subscriber):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(subscriber)
            if len(self._subscribers[event_type]) == 0:
                del self._subscribers[event_type]

    def publish(self, event_type, *args, **kwargs):
        if event_type in self._subscribers:
            for subscriber in self._subscribers[event_type]:
                subscriber(*args, **kwargs)


class Subscriber:
    def __init__(self, name):
        self.name = name

    def receive(self, message):
        print(f"{self.name} received: {message}")


def main():
    bus = Bus()

    subscriber1 = Subscriber("Subscriber 1")
    subscriber2 = Subscriber("Subscriber 2")

    bus.subscribe("event_type_1", subscriber1)
    bus.subscribe("event_type_1", subscriber2)

    bus.publish("event_type_1", "Hello, world!")

    bus.unsubscribe("event_type_1", subscriber1)

    bus.publish("event_type_1", "Goodbye, world!")
