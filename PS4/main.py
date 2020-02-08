import os
import csv
import subprocess
import re
import random
import numpy as np


def read_in_shakespeare():
    '''Reads in the Shakespeare dataset processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []

    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open('vocab.txt') as f:
        vocab = [line.strip() for line in f]

    with open('play_names.txt') as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    td_matrix = np.zeros((len(vocab), len(document_names)))

    for doc_name, line in line_tuples:
        for word in line:
            td_matrix[vocab_to_id[word], docname_to_id[doc_name]] += 1

    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    '''Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    '''

    # YOUR CODE HERE
    tc_matrix = np.zeros((len(vocab), len(vocab)))
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    # Iterate over all documents and corresponding lines
    for doc, line in line_tuples:
      # Iterate over all words in a line
      for ii in range(len(line)):
        target_word = line[ii]
        target_idx = vocab_to_id[target_word]
        # Iterate over the given window
        for jj in range(max(0, ii - context_window_size), min(ii + context_window_size + 1, len(line))):
          context_word = line[jj]
          context_idx = vocab_to_id[context_word]
          tc_matrix[target_idx][context_idx] += 1
    return tc_matrix


def create_PPMI_matrix(term_context_matrix):
    '''Given a term context matrix, output a PPMI matrix.

    See section 15.1 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: A nxn numpy array, where n is
          the numer of tokens in the vocab.

    Returns: A nxn numpy matrix, where A_ij is equal to the
       point-wise mutual information between the ith word
       and the jth word in the term_context_matrix.
    '''

    f_sum = np.sum(term_context_matrix)
    row_sum = np.sum(term_context_matrix, axis=1) # might need to flip row_sum and col_sum if not working
    col_sum = np.sum(term_context_matrix, axis=0)
    mult_sum = np.outer(row_sum, col_sum) / f_sum**2
    ppmi_matrix = np.maximum(np.log2(np.multiply(term_context_matrix / f_sum, 1 / mult_sum) + 1),
                              np.zeros(np.shape(term_context_matrix)))
    return ppmi_matrix


def create_tf_idf_matrix(term_document_matrix):
    '''Given the term document matrix, output a tf-idf weighted version.

    See section 15.2.1 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    '''
    # YOUR CODE HERE
    num_documents = np.size(term_document_matrix, axis=1)
    tf_matrix = np.log10(term_document_matrix + 1) + 1
    idf_matrix = np.log10(num_documents / np.sum(np.heaviside(term_document_matrix, 0), axis=1))
    tf_idf_matrix = (tf_matrix.T * idf_matrix).T
    return tf_idf_matrix



def compute_cosine_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def compute_jaccard_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    return np.sum(np.minimum(vector1, vector2)) / np.sum(np.maximum(vector1, vector2))


def compute_dice_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    j = compute_jaccard_similarity(vector1, vector2)
    return 2 * j / (1 + j)


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    ''' Ranks the similarity of all of the plays to the target play.

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

    Inputs:
      target_play_index: The integer index of the play we want to compare all others against.
      term_document_matrix: The term-document matrix as a mxn numpy array.
      similarity_fn: Function that should be used to compared vectors for two
        documents. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
      A length-n list of integer indices corresponding to play names,
      ordered by decreasing similarity to the play indexed by target_play_index
    '''

    target_play_vector = term_document_matrix[:, target_play_index] # might need different axis
    target_similarity = lambda x: similarity_fn(target_play_vector, x)

    return np.argsort(np.apply_along_axis(target_similarity, 0, term_document_matrix)) # Might need different axis


def rank_words(target_word_index, matrix, similarity_fn):
    ''' Ranks the similarity of all of the words to the target word.

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
      similarity_fn: Function that should be used to compared vectors for two word
        ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
    '''

    target_play_vector = matrix[target_word_index, :]  # might need different axis
    target_similarity = lambda x: similarity_fn(target_play_vector, x)

    return np.argsort(np.apply_along_axis(target_similarity, 1, matrix))


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    random_idx = random.randint(0, len(document_names) - 1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, td_matrix, sim_fn)
        for idx in range(0, 10):
            doc_id = ranks[idx]
            print('%d: %s' % (idx + 1, document_names[doc_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (
        word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))
