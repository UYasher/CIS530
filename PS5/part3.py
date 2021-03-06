from pymagnitude import *
from itertools import combinations
from prettytable import PrettyTable
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from PS4.main import create_PPMI_matrix
import random
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier


def load_input_file(file_path):
    """
    Loads the input file to two dictionaries
    :param file_path: path to an input file
    :return: 2 dictionaries:
    1. Dictionary, where key is a target word and value is a list of paraphrases
    2. Dictionary, where key is a target word and value is a number of clusters
    """
    word_to_paraphrases_dict = {}
    word_to_k_dict = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            target_word, k, paraphrases = line.split(' :: ')
            word_to_k_dict[target_word] = int(k)
            word_to_paraphrases_dict[target_word] = paraphrases.split()

    return word_to_paraphrases_dict, word_to_k_dict


def load_output_file(file_path):
    """
    :param file_path: path to an output file
    :return: A dictionary, where key is a target word and value is a list of list of paraphrases
    """
    clusterings = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            target_word, _, paraphrases_in_cluster = line.strip().split(' :: ')
            paraphrases_list = paraphrases_in_cluster.strip().split()
            if target_word not in clusterings:
                clusterings[target_word] = []
            clusterings[target_word].append(paraphrases_list)

    return clusterings


def write_to_output_file(file_path, clusterings):
    """
    Writes the result of clusterings into an output file
    :param file_path: path to an output file
    :param clusterings:  A dictionary, where key is a target word and value is a list of list of paraphrases
    :return: N/A
    """
    with open(file_path, 'w') as fout:
        for target_word, clustering in clusterings.items():
            for i, cluster in enumerate(clustering):
                fout.write(f'{target_word} :: {i + 1} :: {" ".join(cluster)}\n')
        fout.close()


def get_paired_f_score(gold_clustering, predicted_clustering):
    """
    :param gold_clustering: gold list of list of paraphrases
    :param predicted_clustering: predicted list of list of paraphrases
    :return: Paired F-Score
    """
    gold_pairs = set()
    for gold_cluster in gold_clustering:
        for pair in combinations(gold_cluster, 2):
            gold_pairs.add(tuple(sorted(pair)))

    predicted_pairs = set()
    for predicted_cluster in predicted_clustering:
        for pair in combinations(predicted_cluster, 2):
            predicted_pairs.add(tuple(sorted(pair)))

    overlapping_pairs = gold_pairs & predicted_pairs

    precision = 1. if len(predicted_pairs) == 0 else float(len(overlapping_pairs)) / len(predicted_pairs)
    recall = 1. if len(gold_pairs) == 0 else float(len(overlapping_pairs)) / len(gold_pairs)
    paired_f_score = 0. if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return paired_f_score


def evaluate_clusterings(gold_clusterings, predicted_clusterings):
    """
    Displays evaluation scores between gold and predicted clusterings
    :param gold_clusterings: dictionary where key is a target word and value is a list of list of paraphrases
    :param predicted_clusterings: dictionary where key is a target word and value is a list of list of paraphrases
    :return: N/A
    """
    target_words = set(gold_clusterings.keys()) & set(predicted_clusterings.keys())

    if len(target_words) == 0:
        print('No overlapping target words in ground-truth and predicted files')
        return None

    paired_f_scores = np.zeros((len(target_words)))
    ks = np.zeros((len(target_words)))

    table = PrettyTable(['Target', 'k', 'Paired F-Score'])
    for i, target_word in enumerate(target_words):
        paired_f_score = get_paired_f_score(gold_clusterings[target_word], predicted_clusterings[target_word])
        k = len(gold_clusterings[target_word])
        paired_f_scores[i] = paired_f_score
        ks[i] = k
        table.add_row([target_word, k, f'{paired_f_score:0.4f}'])

    average_f_score = np.average(paired_f_scores, weights=ks)
    print(table)
    print(f'=> Average Paired F-Score:  {average_f_score:.4f}')


# TASK 2.1
def cluster_random(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases randomly
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    clusterings = {}

    random.seed("123")

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]
        # TODO: Implement

        # Give each cluster at least one word
        x = random.sample(paraphrase_list, k)
        clusters = []
        for i in range(k):
            clusters.append([x[i]])

        # Assign the remaining words to clusters at random
        ys = [word for word in paraphrase_list if word not in x]
        for y in ys:
            i = random.randint(0, k-1)
            clusters[i].append(y)

        clusterings[target_word] = clusters

    return clusterings


# TASK 2.2
def cluster_with_sparse_representation(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases using sparse vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    vectors = Magnitude("vectors/coocvec-1000mostfreq-window-7.magnitude")
    clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]

        x = vectors.query(paraphrase_list)
        x = np.maximum(x, np.zeros(np.shape(x)))
        x = create_PPMI_matrix(x)
        # x = PCA(n_components=min(np.size(x, axis=0), 15)).fit_transform(x)
        # clusters = KMeans(n_clusters=k).fit(x)
        clusters = SpectralClustering(n_clusters=k).fit(x)
        # clusters = AgglomerativeClustering(n_clusters=k).fit(x)
        # clusters = DBSCAN().fit(x)
        labels = clusters.labels_

        clusterings[target_word] = []
        for ii in range(k):
            words = []
            for idx, num in enumerate(labels):
                if num == ii:
                    words.append(paraphrase_list[idx])
            if len(words) > 0:
                clusterings[target_word].append(words)
    return clusterings


# TASK 2.3
def cluster_with_dense_representation(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases using dense vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    vectors = Magnitude("vectors/GoogleNews-vectors-negative300.filter.magnitude")
    # vectors = Magnitude("vectors/glove.840B.300d.magnitude")
    # vectors = Magnitude("vectors/glove.6B.300d.magnitude")
    # vectors = Magnitude("vectors/crawl-300d-2M.magnitude")
    # vectors = Magnitude("vectors/GoogleNews-retrofit-wordnet.magnitude")
    dense_clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]

        x = vectors.query(paraphrase_list)
        # x = PCA(n_components=min(np.size(x, axis=0), 15)).fit_transform(x)
        # labels = KMeans(n_clusters=k).fit_predict(x)
        labels = AgglomerativeClustering(n_clusters=k, linkage='single').fit_predict(x)
        # labels = SpectralClustering(n_clusters=k).fit_predict(x)

        dense_clusterings[target_word] = []
        for ii in range(k):
            words = []
            for idx, num in enumerate(labels):
                if num == ii:
                    words.append(paraphrase_list[idx])
            if len(words) > 0:
                dense_clusterings[target_word].append(words)

    return dense_clusterings


# TASK 2.4
def cluster_with_no_k(word_to_paraphrases_dict):
    """
    Clusters paraphrases using any vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    vectors = Magnitude("vectors/crawl-300d-2M.magnitude")
    clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]

        '''
        # Calculate silhouette scores for different k
        x = vectors.query(paraphrase_list)
        range_n_clusters = [x for x in range(1,40,1)]

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(x)

            silhouette_avg = silhouette_score(x, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
      '''

        # Baseline Method (k=5)
        k = 5
        x = vectors.query(paraphrase_list)
        kmeans = KMeans(n_clusters=min(k, len(x))).fit(x)

        # print("kmeans.labels_")
        # print(kmeans.labels_)
        words = [[], [], [], [], []]
        for i in range(len(paraphrase_list)):
                words[kmeans.labels_[i]].append(paraphrase_list[i])
        clusterings[target_word] = words

        # K-means using silhouettes
        # models = []
        # silhouettes = []
        # x = vectors.query(paraphrase_list)
        # for k in range(20):
        #     models.append(KMeans(n_clusters=max(min(k+2, len(x)-1), 2)).fit(x))
        #     try:
        #         silhouettes.append(silhouette_score(x, models[k].labels_))
        #     except:
        #         silhouettes.append(999)
        #
        # max_k = np.argmin(silhouettes)
        # # print("max_k: " + str(max_k))
        # # print("len(models): " + str(len(models)))
        # # print("len(models[int(max_k)].labels_): " + str(len(models[int(max_k)].labels_)))
        # # print("len(paraphrase_list):" + str(len(paraphrase_list)))
        #
        # words = []
        # for k in range(max_k+2):
        #     words.append([])
        # for i in range(len(paraphrase_list)-1):
        #     # print(i)
        #     words[models[int(max_k)].labels_[i]].append(paraphrase_list[i])
        # clusterings[target_word] = words

    return clusterings

# No k clustering
word_to_paraphrases_dict, _ = load_input_file('data/dev_input.txt')
gold_clusterings = load_output_file('data/dev_output.txt')
predicted_clusterings = cluster_with_no_k(word_to_paraphrases_dict)
evaluate_clusterings(gold_clusterings, predicted_clusterings)
write_to_output_file('dev_output_nok.txt', predicted_clusterings)

word_to_paraphrases_dict, _ = load_input_file('data/test_nok_input.txt')
predicted_clusterings = cluster_with_no_k(word_to_paraphrases_dict)
write_to_output_file('test_nok_output.txt', predicted_clusterings)


# word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/test_input.txt')
# predicted_clusterings = cluster_with_dense_representation(word_to_paraphrases_dict, word_to_k_dict)
# write_to_output_file('test_output_leaderboard.txt', predicted_clusterings)
