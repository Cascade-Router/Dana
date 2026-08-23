def fibonacci_up_to(limit: int) -> list[int]:
    sequence = []
    a, b = 0, 1
    while a <= limit:
        sequence.append(a)
        a, b = b, a + b
    return sequence


if __name__ == "__main__":
    fib_seq = fibonacci_up_to(100)
    print("Fibonacci sequence up to 100:")
    print(fib_seq)
