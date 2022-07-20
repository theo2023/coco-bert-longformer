import torch, time, os, sys, random, argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from metrics import compute_metrics
from dataset import *
from model import get_model
from constants import *

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)

    torch.manual_seed(args.seed)
    classifier = args.model
    torch.cuda.set_device(gpu)
    classifier.to(gpu)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    # classifier = DistributedDataParallel(classifier, device_ids=[gpu], find_unused_parameters=False)
    classifier = DistributedDataParallel(classifier, device_ids=[gpu], find_unused_parameters=True) # only for Longformer

    train_df = retrieve_train_data()
    valid_df = retrieve_valid_data()
    train_data = CocoDataset(train_df)
    valid_data = CocoDataset(valid_df)
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_data, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, sampler=valid_sampler)

    patience = 0
    best_valid_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        if patience >= TOLERANCE and gpu == 0:
            print(f"Validation F1 did not improve for {TOLERANCE} epochs. Terminating training.")
            break

        if gpu == 0:
            start = time.time()

        classifier.train()
        train_loss = 0.0
        predictions = []
        gold_labels = []

        for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(train_loader):
            sequence = sequence.to(gpu, non_blocking=True)
            attention_masks = attention_masks.to(gpu, non_blocking=True)
            token_type_ids = token_type_ids.to(gpu, non_blocking=True)
            labels = labels.to(gpu, non_blocking=True)

            # classifier inherits from nn.Module, so this is a call to forward()
            loss, prediction = classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
            loss /= ACCUM_ITERS
            train_loss += loss.item()
            prediction = torch.argmax(prediction, dim=-1)

            loss.backward()
            clip_grad_norm_(classifier.parameters(), 1.0)

            if ((batch_idx + 1) % ACCUM_ITERS == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            predictions.extend(prediction)
            gold_labels.extend(labels)

        train_loss = train_loss / len(train_loader)
        train_precision, train_recall, train_f1, train_acc = compute_metrics(predictions, gold_labels)

        classifier.eval()
        valid_loss = 0.0
        predictions = []
        gold_labels = []

        with torch.no_grad():
            for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(valid_loader):
                sequence = sequence.to(gpu, non_blocking=True)
                attention_masks = attention_masks.to(gpu, non_blocking=True)
                token_type_ids = token_type_ids.to(gpu, non_blocking=True)
                labels = labels.to(gpu, non_blocking=True)

                loss, prediction = classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
                valid_loss += loss.item()
                prediction = torch.argmax(prediction, dim=-1)

                predictions.extend(prediction)
                gold_labels.extend(labels)
        
        valid_loss = valid_loss / len(valid_loader)
        valid_precision, valid_recall, valid_f1, valid_acc = compute_metrics(predictions, gold_labels)

        if gpu == 0:
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                patience = 0 # reset
                print(f"New best validation F1 of {valid_f1:.3f}. Saving model.")
                torch.save(classifier.module.state_dict(), args.path)
            else:
                patience += 1

            end = time.time()
            hours, rem = divmod(end - start, 3600)
            min, sec = divmod(rem, 60)

            print(f"Epoch {epoch + 1}: train_loss: {train_loss:.3f} train_precision: {train_precision:.3f} train_recall: {train_recall:.3f} train_f1: {train_f1:.3f} train_acc: {train_acc:.3f}")
            print(f"\t valid_loss: {valid_loss:.3f} valid_precision: {valid_precision:.3f} valid_recall: {valid_recall:.3f} valid_f1: {valid_f1:.3f} valid_acc: {valid_acc:.3f}")
            print("\t {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(min), sec))

    dist.destroy_process_group()
    sys.exit()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"Effective batch size: {BATCH_SIZE * ACCUM_ITERS} Learning rate: {LEARNING_RATE}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=NUM_GPUS, type=int)
    parser.add_argument('--nr', default=0, type=int)
    parser.add_argument('--path', default=".", type=str)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "8888"

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    args.model = get_model()
    mp.spawn(train, nprocs=args.gpus, args=(args,))