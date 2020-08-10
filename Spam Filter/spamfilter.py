import re
import sys
import random
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


def Sigmoid(x):
    """
    Support function for calculating Sigmoid of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def Diff_Sigmoid(x):
    """
    Support function for calculating Differentiation of Sigmoid of x.
    """
    return np.exp(-x) / (1.0 + np.exp(-x))**2


def Softmax(x):
    """
    Support function for calculating Softmax of x.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def Preprocess(datafile, removewordsfile):
    """
    Loads the data from the datafile, tokenizes it, removes the words from
    removewordsfile, applies Porter Stemming, finds the 500 most repeated words,
    uses them to do one-hot encoding of the tokens, splits the dataset into
    training and testing sets, returns them.
    """
    labels = []
    messages = []
    with open(datafile, 'r') as f:
        for line in f.readlines():
            label, message = line.split("\t")
            labels.append(label)
            messages.append(message)

    labels = np.array(labels)
    messages = np.array(messages)
    tokens = [[
        *filter(
            lambda x: x is not None and (len(x) > 1 or re.match(r"\d{1}", x)),
            re.split(r"[ \t\n\.,:;\-'\(\)\\/!\"\'\“#&%*\+\…<>=_?]+|(\d+)", i))
    ] for i in messages]

    common_words = []
    with open(removewordsfile, 'r') as f:
        for i in f.readlines():
            common_words.append(i[:-1])

    tokens = [[j for j in i if j not in common_words] for i in tokens]

    # Apply Porter Stemming.
    stemmer = PorterStemmer("ORIGINAL_ALGORITHM")
    tokens = [[stemmer.stem(j) for j in i] for i in tokens]
    # One Hot Encoding.
    unique_tokens, count = np.unique(np.hstack(tokens), return_counts=True)
    unique_tokens, _ = zip(
        *sorted(zip(unique_tokens, count), key=lambda x: x[1], reverse=True))
    unique_tokens = unique_tokens[:500]
    # Encode labels.
    labels = [0 if label == "ham" else 1 for label in labels]

    tokens = [[1 if token in row else 0 for token in unique_tokens]
              for row in tokens]

    train, test = train_test_split(list(range(len(labels))), test_size=0.2)
    train_labels = [labels[i] for i in train]
    train_tokens = [tokens[i] for i in train]
    test_labels = [labels[i] for i in test]
    test_tokens = [tokens[i] for i in test]

    return train_labels, train_tokens, test_labels, test_tokens, unique_tokens


def DataLoader(datafile, removewordsfile, batch_size):
    """
    Loads the dataset from the given datafile and generates tokens. The
    processed dataset is then divided into mini-batches each of size batch_size
    Returns the mini-batches, test sets and frequent tokens.
    """
    # Load the dataset into train and test sets.
    train_labels, train_tokens, test_labels, test_tokens, freq_tokens = Preprocess(
        datafile, removewordsfile)
    train_labels = np.array(train_labels)
    train_tokens = np.array(train_tokens)
    test_labels = np.array(test_labels)
    test_tokens = np.array(test_tokens)

    # Create a dataset with train labels and tokens and create batches.
    train_dataset = list(zip(train_labels, train_tokens))
    train_batches = [
        train_dataset[batch_size * i:batch_size * (i + 1)]
        for i in range(len(train_dataset) // batch_size)
    ]
    train_batches.append(
        train_dataset[batch_size * (len(train_dataset) // batch_size):])
    # Return batches and test set and most frequent tokens.
    return train_batches, train_labels, train_tokens, test_labels, test_tokens, freq_tokens


def WeightInitializer(layers):
    """
    Initialize the weights using the layers input variable, returns a list with
    weights initialized to random numbers between 0 and 1.
    Layers input variable contains the number of nodes in the layers including
    input, hidden and output layers.
    """
    return [
        np.array([[random.random() for _ in range(enumi)]
                  for j in range(layers[i + 1])])
        for i, enumi in enumerate(layers[:-1])
    ], [
        np.array([random.random() for _ in range(enumi)])
        for i, enumi in enumerate(layers[1:])
    ]


def ForwardPass(x, weights, biases, layerfunc, outfunc, X, S):
    """
    Calculates the output of neural network given an input
    """
    X.append(x)
    for i, enumi in enumerate(weights[:-1]):
        S.append(enumi @ x + biases[i])
        x = list(map(layerfunc, S[-1]))
        X.append(np.array(x))
    S.append(weights[-1] @ x + biases[-1])
    x = list(map(outfunc, S[-1]))
    X.append(np.array(x))


def BackwardPass(X, S, labels, weights, biases, layer_difffunc, learning_rate):
    """
    Using the values claculated by layer, we find the final error and propagate
    it backwards.
    """
    weight_updates = np.copy(weights)
    biases_updates = np.copy(biases)
    for x, s, actual in zip(X, S, labels):
        # We can directly use this, since the derivative turns out to be as
        # such.
        delta = [np.array([[x[-1][0] - actual], [x[-1][0] - 1 + actual]])]
        for i, weight in reversed(list(enumerate(weights[1:]))):
            delta.append((delta[-1] * weight * layer_difffunc(s[i])).T)
            delta[-1] = np.array([[sum(i)] for i in delta[-1]])
        delta = list(reversed(delta))
        # Get the updates of weights and biases.
        for i, _ in enumerate(weights):
            weight_updates[i] -= (learning_rate * x[i] * delta[i])
            biases_updates[i] -= np.reshape(learning_rate * delta[i],
                                            (delta[i].shape[0], ))

    # Updating the weights and biases.
    for i, _ in enumerate(weights):
        weights[i] = weight_updates[i]
        biases[i] = biases_updates[i]


def train(no_epochs, batches, train_labels, train_tokens, weights, biases,
          rate, test_labels, test_tokens):
    """
    Trains the weights using the batches given.
    """
    for epoch in range(no_epochs):
        for batch in batches:
            X = []
            S = []
            labels = []
            for label, tokens in batch:
                a = []
                b = []
                ForwardPass(tokens, weights, biases, Sigmoid, Softmax, a, b)
                X.append(a)
                S.append(b)
                labels.append(label)
            BackwardPass(X, S, labels, weights, biases, Diff_Sigmoid, rate)

        _, test_pre_spams, test_correct = test(test_labels, test_tokens,
                                               weights, biases, 0.5)
        _, train_pre_spams, train_correct = test(train_labels, train_tokens,
                                                 weights, biases, 0.5)
        try:
            print(
                "epoch: ",
                epoch,
                "\t Train Error: ",
                1 - train_correct / train_pre_spams,
                "\t Test Error: ",
                1 - test_correct / test_pre_spams,
                sep='')
        except Exception as e:
            print(
                "epoch: ",
                epoch,
                "\t Train Error: ",
                1,
                "\t Test Error: ",
                1,
                sep='')


def test(test_labels, test_tokens, weights, biases, threshold, print_info=0):
    """
    Finds the accuracies given the weights and test dataset.
    """
    total_spams = 0
    pre_spams = 0
    correct_predictions = 0
    for label, tokens in zip(test_labels, test_tokens):
        X = []
        S = []
        ForwardPass(tokens, weights, biases, Sigmoid, Softmax, X, S)
        if label == 1:
            total_spams += 1
        if X[-1][0] > threshold:
            pre_spams += 1
            if label == 1:
                correct_predictions += 1

    if print_info:
        print("Total Spams:", total_spams)
        print("Spams Predicted:", pre_spams)
        print("Correct Predictions:", correct_predictions)
        print("Accuracy:", correct_predictions / pre_spams * 100)
    return total_spams, pre_spams, correct_predictions


if __name__ == "__main__":
    TRAIN_BATCHES, TRAIN_LABELS, TRAIN_TOKENS, TEST_LABELS, TEST_TOKENS, FREQ_TOKENS = DataLoader(
        "dataset/Assignment_4_data.txt",
        "dataset/NLTK's list of english stopwords", 10)
    WEIGHTS, BIASES = WeightInitializer(
        [500, int(sys.argv[1]), int(sys.argv[2]), 2])
    # WEIGHTS, BIASES = WeightInitializer([500, 10, 5, 2])
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 10
    train(NUM_EPOCHS, TRAIN_BATCHES, TRAIN_LABELS, TRAIN_TOKENS, WEIGHTS,
          BIASES, LEARNING_RATE, TEST_LABELS, TEST_TOKENS)

    print("\nFinal Report:")
    test(TEST_LABELS, TEST_TOKENS, WEIGHTS, BIASES, 0.5, print_info=1)
