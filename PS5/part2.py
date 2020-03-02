import pandas as pd
import scipy.stats as stats
from pymagnitude import *


def main():
    vectors = Magnitude('vectors/glove.840B.300d.magnitude')
    df = pd.read_csv('data/SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)
        print(f'{word1},{word2},{score},{similarity_score:.4f}')

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    print(f'Correlation = {correlation}, P Value = {p_value}')

    df['vector_scores'] = vector_scores

    simlex_sorted = sorted(df.values.tolist(), key=lambda row1: -1 * row1[2])
    print("According to simlex, the most similar are: ")
    print(simlex_sorted[0])

    vectors_sorted = sorted(df.values.tolist(), key=lambda row1: -1 * row1[3])
    print("According to GoogleNew-vectors, the most similar are:")
    print(vectors_sorted[0])

    simlex_sorted = sorted(df.values.tolist(), key=lambda row1: row1[2])
    print("According to simlex, the least similar are: ")
    print(simlex_sorted[0])

    vectors_sorted = sorted(df.values.tolist(), key=lambda row1: row1[3])
    print("According to GoogleNew-vectors, the least similar are:")
    print(vectors_sorted[0])


if __name__ == '__main__':
    main()
