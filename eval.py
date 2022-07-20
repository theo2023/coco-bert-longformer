import torch, os, random, argparse
import numpy as np

from torch.utils.data import DataLoader
from metrics import compute_metrics
from dataset import *
from model import get_model
from constants import *

def test(classifier, test_loader, device):
    classifier.eval()
    test_loss = 0.0
    predictions = []
    gold_labels = []

    with torch.no_grad():
        for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(test_loader):
            sequence = sequence.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            loss, prediction = classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
            test_loss += loss.item()
            prediction = torch.argmax(prediction, dim=-1)

            predictions.extend(prediction)
            gold_labels.extend(labels)
        
    test_loss = test_loss / len(test_loader)
    test_precision, test_recall, test_f1, test_acc = compute_metrics(predictions, gold_labels)

    print(f"test_loss: {test_loss:.3f} test_precision: {test_precision:.3f} test_recall: {test_recall:.3f} test_f1: {test_f1:.3f} test_acc: {test_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--path', default=".", type=str)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    classifier = get_model()

    print(80 * "=")
    print("TESTING")
    print(80 * "=")

    test_df = retrieve_test_data()
    test_data = CocoDataset(test_df)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Restoring the best model weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.load_state_dict(torch.load(args.path), strict=False)
    classifier.to(device)
    print("Final evaluation on test set")
    test(classifier, test_loader, device)