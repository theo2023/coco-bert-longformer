from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(predicted_labels, gold_labels):
    predicted_labels = [label.item() for label in predicted_labels]
    gold_labels = [label.item() for label in gold_labels]

    assert len(predicted_labels) == len(gold_labels)

    precision = precision_score(gold_labels, predicted_labels, zero_division=0)
    recall = recall_score(gold_labels, predicted_labels, zero_division=0)
    f1 = f1_score(gold_labels, predicted_labels, zero_division=0)
    accuracy = accuracy_score(gold_labels, predicted_labels) 

    return precision, recall, f1, accuracy