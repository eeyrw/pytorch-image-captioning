import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder
from utils.decoding_utils import greedy_decoding
from tqdm import tqdm


def evaluate(subset, encoder, decoder, config, device):
    """Evaluates (BLEU score) caption generation model on a given subset.

    Arguments:
        subset (Flickr8KDataset): Train/Val/Test subset
        encoder (nn.Module): CNN which generates image features
        decoder (nn.Module): Transformer Decoder which generates captions for images
        config (object): Contains configuration for the evaluation pipeline
        device (torch.device): Device on which to port used tensors
    Returns:
        bleu (float): BLEU-{1:4} scores performance metric on the entire subset - corpus bleu
    """
    batch_size = config["batch_size"]["eval"]
    max_len = config["max_len"]
    bleu_w = config["bleu_weights"]

    # Mapping from vocab index to string representation
    idx2word = subset._idx2word
    # Ids for special tokens
    sos_id = subset._start_idx
    eos_id = subset._end_idx
    pad_id = subset._pad_idx

    references_total = []
    predictions_total = []

    print("Evaluating model.")
    for x_img, y_caption in tqdm(subset.inference_batch(batch_size)):
        x_img = x_img.to(device)

        # Extract image features
        img_features = encoder(x_img)
        img_features = img_features.view(
            img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.detach()

        # Get the caption prediction for each image in the mini-batch
        predictions = greedy_decoding(
            decoder, img_features, sos_id, eos_id, pad_id, idx2word, max_len, device)
        references_total += y_caption
        predictions_total += predictions

    # Evaluate BLEU score of the generated captions
    bleu_1 = corpus_bleu(references_total, predictions_total,
                         weights=bleu_w["bleu-1"]) * 100
    bleu_2 = corpus_bleu(references_total, predictions_total,
                         weights=bleu_w["bleu-2"]) * 100
    bleu_3 = corpus_bleu(references_total, predictions_total,
                         weights=bleu_w["bleu-3"]) * 100
    bleu_4 = corpus_bleu(references_total, predictions_total,
                         weights=bleu_w["bleu-4"]) * 100
    bleu = [bleu_1, bleu_2, bleu_3, bleu_4]
    return bleu


def main():

    # Load the pipeline configuration file
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    """Performs the training of the model.

    Arguments:
        config (object): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

   # Define dataloader hyper-parameters
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "num_workers": 1,
        "drop_last": True
    }

    # Create dataloaders
    valid_set = Flickr8KDataset(
        config, config["split_save"]["validation"], training=False)

    #######################
    # Set up the encoder
    #######################
    # Download pretrained CNN encoder
    encoder = models.resnet50(pretrained=True)
    # Extract only the convolutional backbone of the model
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    encoder = encoder.to(device)
    # Freeze encoder layers
    for param in encoder.parameters():
        param.requires_grad = False

    ######################
    # Set up the decoder
    ######################
    # Instantiate the decoder
    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)

    if config["checkpoint"]["load"]:
        checkpoint_path = config["checkpoint"]["path"]
        decoder.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        # Evaluate model performance on subsets
        train_bleu = evaluate(valid_set, encoder, decoder, config, device)

        # Log the evaluated BLEU score
        for i, t_b in enumerate(train_bleu):
            print(f"Train/BLEU-{i+1}", t_b, train_bleu)

if __name__ == "__main__":
    main()

