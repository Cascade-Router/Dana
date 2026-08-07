class EventBus:
    def __init__(self):
        self.subscribers = []

    def publish(self, event):
        for subscriber in list(self.subscribers):
            if isinstance(subscriber, tuple):
                topic, callback = subscriber
                if topic == event:
                    callback(event)
            else:
                subscriber(event)

    def subscribe(self, topic_or_callback, callback=None):
        if callback is None:
            callback = topic_or_callback
            self.subscribers.append(callback)
            return

        subscribers_key = (topic_or_callback, callback)
        self.subscribers.append(subscribers_key)

    def unsubscribe(self, topic_or_callback, callback=None):
        if callback is None:
            callback = topic_or_callback
            if callback in self.subscribers:
                self.subscribers.remove(callback)
                return
            raise ValueError("Subscriber not found")

        subscribers_key = (topic_or_callback, callback)
        if subscribers_key in self.subscribers:
            self.subscribers.remove(subscribers_key)
            return
        raise ValueError("Subscriber not found")
