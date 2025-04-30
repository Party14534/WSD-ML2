import html.parser
import sys


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


# Only care about specific tags
def valid_tag(tag: str):
    match(tag):
        case 'answer' | 'instance' | 's' | 'head' | 'context':
            return True

    return False


# Exit if there arent enough arguments
if len(sys.argv) < 3:
    print("Invalid number of arguments")
    print(len(sys.argv))
    exit(1)

# Load files
guesses_filename = sys.argv[1]
answers_filename = sys.argv[2]
guess_file = open(guesses_filename, 'r')
answer_file = open(answers_filename, 'r')

guess_data = guess_file.read()
answer_data = answer_file.read()

guesses = guess_data.split('\n')
answers = []

# Parse the answer file to get the answers
answer_parser = MyParser()
answer_parser.feed(answer_data)
for tag in answer_parser.values:
    if tag != 'answer' and tag != '\n':
        answers.append(tag)

# Go through the guesses to see if they're the same as the answers
# and create matrix
correct_count = 0
count = len(answers)
correct_product = 0  # Guessed product and was product
correct_phone = 0  # Guessed phone and was phone
incorrect_product = 0  # Guessed phone but was product
incorrect_phone = 0  # Guessed product but was phone

for i, guess in enumerate(guesses):
    # The last guess is an empty string
    if guess == '':
        continue
    if guess == answers[i]:
        correct_count += 1

        if guess == "phone":
            correct_phone += 1
        else:
            correct_product += 1
    elif guess == "phone":
        incorrect_product += 1
    else:
        incorrect_phone += 1


# Calculate the accuracy
accuracy = float(correct_count) / float(count)
print(f"Accuracy: {accuracy * 100}%")

# Print out the matrix
print("Matrix: ")
print("          product | phone |")
print("        -------------------")
print(f"product |   {correct_product}   |   {incorrect_phone}  |")
print("        -------------------")
print(f"phone   |   {incorrect_product}   |   {correct_phone}  |")
