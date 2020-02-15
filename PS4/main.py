import os
import csv
import subprocess
import re
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

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


def read_in_shakespeare_characters(include_act=False):
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
        character_names = []
        for row in csv_reader:
            character_name = row[4]
            if include_act:
                #print("row[3]: " + str(row[3]))
                character_name += row[3].split(".")[0]
                #print(character_name)
            character_names.append(character_name)
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((character_name, line_tokens))

    with open('vocab.txt') as f:
        vocab = [line.strip() for line in f]

    character_names = list(set(character_names))

    return tuples, character_names, vocab


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

    target_play_vector = term_document_matrix[:, target_play_index]  # might need different axis
    target_similarity = lambda x: -1 * similarity_fn(target_play_vector, x)
    outputs = np.apply_along_axis(target_similarity, 0, term_document_matrix)
    output_indices = np.argsort(outputs)
    print(outputs[output_indices][1:11])
    return output_indices  # Might need different axis


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
    target_similarity = lambda x: -1 * similarity_fn(target_play_vector, x)
    outputs = np.apply_along_axis(target_similarity, 1, matrix)
    output_indices = np.argsort(outputs)
    print(outputs[output_indices][1:11])
    return output_indices


if __name__ == '__main__':
    '''
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=3)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    random_idx = 9
    # random_idx = random.randint(0, len(document_names) - 1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, tf_idf_matrix, sim_fn)
        for idx in range(0, 10):
            doc_id = ranks[idx]
            print('%d: %s' % (idx + 1, document_names[doc_id]))

    word = 'death'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (
        word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))

    word = 'death'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))
    '''

    '''
    # Insult Analysis
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]

    word = 'loon'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (
            word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))

    word = 'loon'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx + 1, vocab[word_id]))

    '''

    # Character Analysis
    tuples, character_names, vocab = read_in_shakespeare_characters(include_act=True)

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, character_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    character_to_index = dict(zip(character_names, range(0, len(character_names))))
    #character = 'DEMETRIUS1'

    # character_id = character_to_index[character]
    # similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    # for sim_fn in similarity_fns:
    #     print('\nThe 10 most similar characters to "%s" using %s are:' % (character_names[character_id], sim_fn.__qualname__))
    #     ranks = rank_plays(character_id, td_matrix, sim_fn)
    #     for idx in range(0, 10):
    #         doc_id = ranks[idx]
    #         print('%d: %s' % (idx + 1, character_names[doc_id]))

    # DYNAMIC CHARACTERS
    comparison_characters = ["DEMETRIUS" + str(i) for i in range(1, 6, 1)]
    #comparison_characters = ["PUCK" + str(i) for i in range(2, 6, 1)]
    #comparison_characters = [ "LYSANDER" + str(i) for i in range(1,6, 1)]
    #comparison_characters = ["ROMEO" + str(i) for i in [1, 2, 3, 5]]
    #comparison_characters = ["HAMLET" + str(i) for i in [1, 2, 3, 4, 5]]
    #comparison_characters = ["KING LEAR" + str(i) for i in [1, 2, 3, 4, 5]]

    # STATIC CHARACTERS
    #comparison_characters = ["THESEUS" + str(i) for i in [1,4,5]]
    #comparison_characters = ["HIPPOLYTA" + str(i) for i in [1, 4, 5]]
    #comparison_characters = ["EGEUS" + str(i) for i in [1, 4]]
    #comparison_characters = ["BENVOLIO" + str(i) for i in [1, 2, 3]]
    #comparison_characters = ["MERCUTIO" + str(i) for i in [1, 2, 3]]
    #comparison_characters = ["HORATIO" + str(i) for i in [1, 3, 4, 5]]
    #comparison_characters = ["HERO" + str(i) for i in [1, 2, 3, 4, 5]]
    #comparison_characters = ["CLAUDIO" + str(i) for i in [1, 2, 3, 4, 5]]

    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    sim_matrices = { sim_fn.__qualname__:[] for sim_fn in similarity_fns }
    for i in range(len(comparison_characters)):
        character = comparison_characters[i]
        print("Conduting comparions for " + character)
        character_vec = get_column_vector(td_matrix, character_to_index[character])
        for sim_fn in similarity_fns:
            print('\nThe character similarities to to "%s" using %s are:' % (
            character, sim_fn.__qualname__))

            sim_matrices[sim_fn.__qualname__].append([])

            for j in range(len(comparison_characters)):
                comparison_character = comparison_characters[j]
                comp_char_vec = get_column_vector(td_matrix, character_to_index[comparison_character])
                similarity = sim_fn(character_vec, comp_char_vec)
                print("His similarity to " + comparison_character + " is: " + str(similarity))
                sim_matrices[sim_fn.__qualname__][i].append(similarity)

    print(sim_matrices)
    for sim_fn in similarity_fns:
        fig, ax = plt.subplots()
        sim_matrix = np.array(sim_matrices[sim_fn.__qualname__])
        im = ax.imshow(sim_matrix)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(comparison_characters)))
        ax.set_yticks(np.arange(len(comparison_characters)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(comparison_characters)
        ax.set_yticklabels(comparison_characters)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(comparison_characters)):
            for j in range(len(comparison_characters)):
                text = ax.text(j, i, round(sim_matrix[i, j], 3),
                               ha="center", va="center", color="w")

        ax.set_title("Character " + sim_fn.__qualname__ + " Similarity Across Acts")
        fig.tight_layout()
        plt.show()

    # td_matrix = td_matrix.T
    # tf_idf_matrix = tf_idf_matrix.T
    # kmeans = KMeans(n_clusters=3)
    # preds = kmeans.fit_predict(td_matrix)
    # group_one = [name for idx, name in enumerate(document_names) if preds[idx] == 0]
    # group_two = [name for idx, name in enumerate(document_names) if preds[idx] == 1]
    # group_three = [name for idx, name in enumerate(document_names) if preds[idx] == 2]
    # print(group_one)
    # print(group_two)
    # print(group_three)
    # pca = PCA(n_components=3)
    # reduced_matrix = pca.fit_transform(td_matrix)
    # kmeans = KMeans(n_clusters=3)
    # preds = kmeans.fit_predict(reduced_matrix)
    # color = ['r', 'g', 'b']
    # colors = [color[idx] for idx in preds]
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(reduced_matrix[:, 0], reduced_matrix[:, 1], reduced_matrix[:, 2], c=colors)
    # plt.title('Principal Components of Document Clusters')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # plt.show()
    # group_one = [name for idx, name in enumerate(document_names) if preds[idx] == 0]
    # group_two = [name for idx, name in enumerate(document_names) if preds[idx] == 1]
    # group_three = [name for idx, name in enumerate(document_names) if preds[idx] == 2]
    # print(group_one)
    # print(group_two)
    # print(group_three)
