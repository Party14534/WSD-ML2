"""
Zachariah Dellimore V00980652

Using scikit learn SVM I was able to improve upon my previous scikit learn
accuracy by ~6% while the pytorch neural network performed ~8% better than my
previous scikit learn models and the models performed much better than the most
frequent sense baseline which was 52.4%

Models used
SVM
    Accuracy: 96.03174603174604%
    Matrix:
              product | phone |
            -------------------
    product |   50   |   1  |
            -------------------
    phone   |   4   |   71  |

    SVM performed better than my previous scikit learn models and it performed
    worse than the Pytorch Neural Network.


Pytorch Neural Network
    Accuracy: 99.20634920634922%
    Matrix:
              product | phone |
            -------------------
    product |   54   |   1  |
            -------------------
    phone   |   0   |   71  |

    The pytorch neural network performed better than every other model I've
    used missing only one classification.
"""

import html.parser
import copy
import string
import sys
import torch
from typing import OrderedDict
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# Pytorch neural network class
class MyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decision:
    feature = ""
    answer = ""
    liklihood = 0.0


# Sentence structure to save important data
class Sentence:
    sentence = ""
    word = ""
    answer = ""


# Parser implementation
class MyParser(html.parser.HTMLParser):
    values = []

    def handle_starttag(self, tag, attrs):
        if valid_tag(tag):
            if tag == 'answer':
                self.values.append(tag)
                self.values.append(attrs[1][1])
            else:
                self.values.append(tag)

    def handle_endtag(self, tag):
        self.values.append(tag)

    def handle_data(self, data):
        self.values.append(data)


sentences = []
features = OrderedDict()
feature_count = {}
words = {}
answer_count = {}
glove_dict = {}
num_answers = 0
model_type = "NaiveBayes"


# Only care about specific tags
def valid_tag(tag: str):
    match(tag):
        case 'answer' | 'instance' | 's' | 'head' | 'context':
            return True

    return False


# Parses the glove file to get vectors
def parse_glove(glove: str):
    lines = glove.split('\n')

    for line in lines:
        words = line.split(' ')
        nums = []
        for num in words[1:]:
            nums.append(float(num))

        glove_dict[words[0]] = nums


# Convert sentences to vectors
def sentence_to_vec(sentence, glove_dict):
    words = sentence.lower().split()
    word_vecs = [glove_dict[word] for word in words if word in glove_dict]

    # Return a zero vector if no words are known
    if not word_vecs:
        return np.zeros(300)
    else:
        word_vecs = np.array(word_vecs)

        # Return the average of the vectors
        return np.mean(word_vecs, axis=0)


def main():
    global model_type

    # Get the model to use
    if len(sys.argv) < 3:
        print("Invalid number of arguments!")
        exit(1)
    elif len(sys.argv) == 4:
        model_type = "SVM"
    elif len(sys.argv) == 5:
        match(sys.argv[4]):
            case "NN" | "SVM":
                model_type = sys.argv[4]
            case _:
                print("Invalid model name")
                exit(1)

    # Load file data
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    glove_filename = sys.argv[3]
    train_file = open(train_filename, 'r')
    test_file = open(test_filename, 'r')
    glove_file = open(glove_filename, 'r')

    train_data = train_file.read()
    test_data = test_file.read()
    glove_data = glove_file.read()

    # Get the glove data
    parse_glove(glove_data)

    # Parse data
    parser = MyParser()
    parser.feed(train_data)

    # Convert the parser data to sentences
    global words
    sentence = Sentence()
    creatingSentence = False
    insideHead = False
    insideContext = False
    insideS = False
    insideAnswer = False
    for tag in parser.values:
        if not creatingSentence:
            if tag == 'instance':
                creatingSentence = True
        else:
            match(tag):
                # The cases are used to tell which tags we are currently
                # inside to know where each string should go
                case 'instance':
                    # Instance tags surround data
                    creatingSentence = False

                    # Filter the sentence to remove unnecessary punctuation
                    # for disambiguation
                    filtered_sentence = sentence.sentence.translate(
                            str.maketrans('', '', string.punctuation))
                    sentence.sentence = filtered_sentence
                    for word in filtered_sentence.split(' '):
                        words[word] = True

                    # append sentence to list of sentences
                    sentences.append(copy.deepcopy(sentence))
                    sentence.sentence = ""
                    sentence.word = ""
                    sentence.answer = ""
                case 'context':
                    insideContext = not insideContext
                case 's':
                    insideS = not insideS
                case 'head':
                    insideHead = not insideHead
                    insideS = not insideS
                case 'answer':
                    insideAnswer = not insideAnswer
                case _:
                    if insideAnswer:
                        # This is where we get the feature
                        sentence.answer = tag
                    elif insideS:
                        # This is where we get the sentence data
                        sentence.sentence += tag
                    elif insideHead:
                        # This is where we get the word to be disambiguated
                        sentence.word = tag
                        sentence.sentence += tag

    # Create test sentences
    test_parser = MyParser()
    test_parser.values = []
    test_parser.feed(test_data)

    # Parse the test data and convert it to sentences
    test_sentences = []
    test_sentence = Sentence()
    test_sentence.sentence = ""
    test_sentence.word = ""
    test_sentence.answer = ""

    # This is the same as parsing the training data above
    for tag in test_parser.values:
        if not creatingSentence:
            if tag == 'instance':
                creatingSentence = True
        else:
            match(tag):
                case 'instance':
                    creatingSentence = False
                    test_sentences.append(copy.deepcopy(test_sentence))
                    test_sentence.sentence = ""
                    test_sentence.word = ""
                    test_sentence.answer = ""
                case 'context':
                    insideContext = not insideContext
                case 's':
                    insideS = not insideS
                case 'head':
                    insideHead = not insideHead
                    insideS = not insideS
                case 'answer':
                    insideAnswer = not insideAnswer
                case _:
                    if insideAnswer:
                        test_sentence.answer = tag
                    elif insideS:
                        test_sentence.sentence += tag
                    elif insideHead:
                        test_sentence.word = tag
                        test_sentence.sentence += tag

    # Get data ready for SVM and NN
    x_train = [sentence_to_vec(s.sentence, glove_dict) for s in sentences]
    y_train = [s.answer for s in sentences]

    x_test = [sentence_to_vec(s.sentence, glove_dict) for s in test_sentences]

    match(model_type):
        case "SVM":
            clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1.0))
            clf.fit(x_train, y_train)

            predictions = clf.predict(x_test)
            for prediction in predictions:
                print(prediction)

        case "NN":
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)

            x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            model = MyNN(len(x_train[0]), 128, 2)
            crit = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # train the model
            for _ in range(20):
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = crit(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # Have the model predict
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            outputs = model(x_test_tensor)
            _, predicted = torch.max(outputs, 1)

            answers = label_encoder.inverse_transform(predicted.numpy())

            for answer in answers:
                print(answer)


if __name__ == "__main__":
    main()
