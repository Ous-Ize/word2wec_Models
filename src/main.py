from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.nlpds.submission.ex1.word2vec.data.dataset import Dataset
from src.nlpds.submission.ex1.word2vec.data.negative import \
    DatasetNegativeSampling, NegativeSampler
from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer, \
    PreTokenizer
from src.nlpds.submission.ex1.word2vec.model.skipgram import \
    SkipGramNegativeSampling


data_root = Path("../data/ex1/")
training_text_file = data_root / "train_enwiki.txt"

# reading the text-file und setting the sentences-corpus
with open(training_text_file, "r") as text:
    corpus = text.read().split("\n")
    corpus_size = len(corpus)

print("corpus of sentences initialised")

model_preTokenizer = PreTokenizer()
model_tokenizer = Tokenizer(dict(), model_preTokenizer)

dataset = Dataset.new(corpus, model_tokenizer, window_size=3, threshold=0.001)
dataset_with_negative_samples = DatasetNegativeSampling.new(corpus,
                                                            model_tokenizer,
                                                            3,
                                                            0.001,
                                                            NegativeSampler.new(dataset.token_frequencies,
                                                           3,
                                                            power=0.75)
                                                            )

print("Data set with negative samples initialised")

vocabulary_size = len(dataset_with_negative_samples.token_counts)
embedding_size = 3

model = SkipGramNegativeSampling(vocabulary_size, embedding_size)

print("model works")

# no need for criterion, loss is in forward computed
criterion = nn.CrossEntropyLoss()

learn_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

num_epochs: int = 200

loss_history = np.array(num_epochs)

for epoch in range(num_epochs):
    train_dataloader: DataLoader = DataLoader(dataset_with_negative_samples, batch_size=50)

    model.train()

    for batch in train_dataloader:
        input_id, context_ids, negativeids = batch   # input_id, context_ids, negative_ids

        # compute prediction
        loss = model(input_id, context_ids, negativeids)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[epoch] = loss

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss_history[epoch]}")


# create models directory
MODEL_PATH = Path("word2wec_models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
MODEL_NAME = "SkipGramNegativeSampling_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
torch.save(model.state_dict(), MODEL_SAVE_PATH)


# visualisation the loss during the training
plt.plot(np.array(range(num_epochs)), loss_history, label="loss")

plt.xlabel("epoch")
plt.ylabel("loss-value")

plt.legend()
plt.show()

