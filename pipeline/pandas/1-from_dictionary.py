import pandas as pd

if __name__ == "__main__":
    data = {"First": [0.0, 0.5, 1.0, 1.5], "Second": ["one", "two", "three", "four"]}
    df = pd.DataFrame(data, index=["A", "B", "C", "D"])
    print(df)
