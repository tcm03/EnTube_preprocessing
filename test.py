from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

if __name__ == "__main__":
    golds = []
    preds = []
    with open("data/EnTube_1h_test.json", "r") as f:
        data = json.load(f)
        for item in data:
            label = int(item["label"])
            preds.append(2)
            golds.append(label)
    print(accuracy_score(golds, preds))
    print(precision_recall_fscore_support(golds, preds, average='macro'))
    print(precision_recall_fscore_support(golds, preds, average='micro'))
    print(precision_recall_fscore_support(golds, preds, average='weighted'))