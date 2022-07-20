from transformers import BertForSequenceClassification, LongformerForSequenceClassification, logging
from constants import NUM_CLASSES

def get_model():
    logging.set_verbosity_error()
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_CLASSES)
    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=NUM_CLASSES)
    return model