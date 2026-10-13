import sys
import pandas as pd


def read_labels(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df.iloc[:, -1]  


def train(labels):
    zeros = (labels == "0").sum()
    ones = (labels == "1").sum()
    return "1" if ones >= zeros else "0"  #


def predict(label, n):
    return [label] * n


def error(pred, true):
    mistakes = 0
    for i in range(len(true)):
        if pred[i] != true.iloc[i]:
            mistakes += 1
    return mistakes / len(true)



if __name__ == "__main__":
    train_in, test_in, train_out, test_out, metrics_out = sys.argv[1:6]

    y_train = read_labels(train_in)
    y_test = read_labels(test_in)

    majority = train(y_train)
    predicted_train = predict(majority, len(y_train))
    predicted_test = predict(majority, len(y_test))

    with open(train_out, "w") as f:
        f.write("\n".join(predicted_train) + "\n")
    with open(test_out, "w") as f:
        f.write("\n".join(predicted_test) + "\n")
    with open(metrics_out, "w") as f:
        f.write(f"error(train): {error(predicted_train, y_train):.6f}\n")
        f.write(f"error(test): {error(predicted_test, y_test):.6f}\n")
