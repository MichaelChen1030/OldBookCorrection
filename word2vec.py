# encoding=utf-8

import re
import string
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
        return out

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# print(make_context_vector(data[0][0], word_to_ix))  # example

if __name__ == '__main__':
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    EMBEDDING_SIZE = 128
    remove = string.punctuation
    remove = remove.replace("-", "").replace("'", "")  # don't remove hyphens and apostrophes
    pattern = r"[{}]".format(remove)  # create the pattern

    with open('1601.txt', 'r') as f:
        text = f.read().replace('\n', ' ')
        raw_text = re.sub(pattern, "", text).split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []

    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    loss_func = nn.CrossEntropyLoss()
    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            log_probs = net(context_var)
            loss = loss_func(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        print(total_loss)

    print(CBOW.embeddings)