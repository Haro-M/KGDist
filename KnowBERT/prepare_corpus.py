
# Set the directory containing your input files
input_dir = 'domain_corpus/'

# Set the directory for your output files
output_dir = 'output_corpus/'

import os
import nltk
nltk.download('punkt')  # Download the Punkt tokenizer if not already present

def sentence_tokenize(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as in_file, open(output_file, 'w', encoding='utf-8') as out_file:
        # Read the input file
        content = in_file.read()

        # Parse the content using NLTK
        sentences = nltk.sent_tokenize(content)

        # Write each sentence to a new line in the output file
        for sent in sentences:
           if len(sent) > 20:
              out_file.write(sent.strip() + '\n')

        # Add a blank line to separate documents
        out_file.write('\n')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        sentence_tokenize(input_file, output_file)
        print(f"Processed {input_file} and saved to {output_file}")
