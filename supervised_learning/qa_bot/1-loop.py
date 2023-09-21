STOP_WORDS = {"exit", "quit", "goodbye", "bye"}


def main() -> None:
    while True:
        in_value = input("Q:").rstrip('\n')
        if in_value.lower() in STOP_WORDS:
            print("A: Goodbye")
            break
        print("A:")


if __name__ == "__main__":
    main()
