import re
import time
import similaripy # for highly optimized nearest neighbors algorithms for sparse matrices
import numpy as np # for ndarray, used interchangeably with standard python list
import scipy.sparse as sps # provides sparse matrices needed for extremely sparse TF/IDF matrix
from collections import Counter # provides 'uniqe' function that doesnt rearrange the elements
from sklearn.feature_extraction.text import CountVectorizer # for quick counting of words occurences

class label_predictor():
    
    def __init__(self, verbose=True, k_neighbors=46):
        '''
        if verbose is set to true, model will print its progress with details
        k_neighbors tells how many neighbors take into consideration for label choice
        '''
        
        self.verbose = verbose
        self.k = k_neighbors
    
    
    def fit(self, dataset, labels):
        '''
        dataset must be a 2-column ndarray, where columns contain title, description
        labels must be an ndarray vector with all the labels separated by space in a string
        '''
        
        assert dataset.shape[0] == labels.shape[0]
        if self.verbose: print("Preprocessing titles", end='')
        dataset[:, 0] = list(map(self.preprocess_polish, dataset[:, 0]))
        if self.verbose: print(", descriptions", end='')
        dataset[:, 1] = list(map(self.preprocess_polish, dataset[:, 1]))
        
        self.dataset = dataset
        self.train_size = dataset.shape[0]
        
        if self.verbose: print(", labels")
        for i, label in enumerate(labels):
            if not isinstance(label, str): labels[i] = str(label)
        
        self.train_labels = labels
                
        all_labels = [t for labels in labels[:self.train_size] for t in labels.split()]
        self.voc = {w for label in all_labels for w in label.split('_') if w!=''}
        word_counts = self.count_words(self.dataset[:, 0], self.dataset[:, 1])
        self.word_counts = 2.8 * word_counts[0] + 0.5 * word_counts[1]
        if self.verbose: print("Shape of words occurence matrix is", self.word_counts.shape)
        if self.verbose: print("DONE!")
    
    

    def predict(self, test_dataset):
        '''
        format of data in testing dataset must be the same as in the training dataset
        returns an ndarray with 5 suggested labels for each row in test_dataset
        '''
        
        if self.verbose: print("Preprocessing titles", end='')
        test_dataset[:, 0] = list(map(self.preprocess_polish, test_dataset[:, 0]))
        if self.verbose: print(", descriptions")
        test_dataset[:, 1] = list(map(self.preprocess_polish, test_dataset[:, 1]))
        
        word_counts = self.count_words(test_dataset[:, 0], test_dataset[:, 1])
        word_counts = 2.8 * word_counts[0] + 0.5 * word_counts[1]
        word_counts = sps.vstack([self.word_counts, word_counts])
        
        tfidf = similaripy.normalization.tfidf(word_counts)
        if self.verbose: print("Shape of a full TF/IDF matrix is", word_counts.shape)
        
        if self.verbose: print("Searching for", self.k, "closest neighbors:"); time.sleep(0.5)
        similarities = similaripy.s_plus( l=0.625, t2=0, t1=0, c=0.65,
                                matrix1=tfidf[self.train_size:], matrix2=tfidf[:self.train_size].T,
                                k=self.k, format_output='csr', verbose=self.verbose)
        
        if self.verbose: print("Dealing with missing neighbors", end='')
        nonzero_rows = np.unique( similarities.nonzero()[0], return_counts=True )
        missing = np.zeros(similarities.shape[0])
        missing[nonzero_rows[0]] = nonzero_rows[1]
        missing_neigh = np.where(missing!=self.k)[0]
        similarities = sps.lil_matrix(similarities)
        replaced = 0
        for i, row in enumerate(missing_neigh):
            j=-1
            limit = self.k - len(similarities.getrow(row).nonzero()[1])
            while j < limit-1:
                j += 1
                if similarities[row, j] != 0: limit+=1; continue
                similarities  [ row, j] = -1
                replaced += 1
        similarities = sps.csr_matrix(similarities)
        if self.verbose: print(", replaced", replaced, "missing!")

        similarities.eliminate_zeros()
        res_dense = similarities.data.reshape(-1, self.k)
        distances = -np.sort( -res_dense )
        indices = np.argsort( -res_dense )
        knn = similarities.nonzero()[1].reshape(-1, self.k)
        knn = np.take_along_axis(knn, indices, 1)
        
        to_extract = 11 # how many true labels extract from neighbours
        true_extracted_labels = np.array([' '.join(x.split()[:to_extract]) for x in self.train_labels])
        found_labels = true_extracted_labels[ knn ]
        
        labels_per_element_limit = 25
        
        labels_per_adv = [ [ len(f.split(' ')) for f in labels ] for labels in found_labels[:, :self.k] ]
        mask = np.linspace(1,0.0005,11)**0.15 # for weighting for being 1-st, 2-nd label of a adv

        dist_per_label = [ np.zeros( sum(adv)) for adv in labels_per_adv ]
        for case, adv in enumerate(labels_per_adv):
            act = 0
            for num_idx, var in enumerate(adv):
                dist_per_label[case][act : act+var] = distances[case, num_idx] * mask[:var]
                act += var
                
        labels_conn = [' '.join(labels).split(' ') for labels in found_labels[:, :self.k]]
        all_points = [ {label: 0 for label in labels} for labels in labels_conn ]
        
        for j, case in enumerate(labels_conn):
            if self.verbose and j%5000==0: print('\r{}{}/{}'.format("Scoring the labels: ", j, len(labels_conn)), end='')
            for i, p in enumerate(case):
                all_points[j][p] += dist_per_label[j][i]**8 #* weight(i, len(case))
        if self.verbose: print('\r{}{}/{}'.format("Scoring the labels: ", len(labels_conn), len(labels_conn)))
        
        if self.verbose: print("Choosing final labels!")
        scores = [sorted( points.items(), key=lambda x: -x[1] )[:labels_per_element_limit] for points in all_points]
        found_labels_list = [ [x[0] for x in labels] for labels in scores ]
        found_labels_rank = [ [x[1] for x in labels] for labels in scores ]
        
        answers = np.array([ self.choose_labels_for_text(labels, rank, text_title, text) for labels, rank,
                            text_title, text in zip( found_labels_list, found_labels_rank,
                            test_dataset[:, 0], test_dataset[:, 1] ) ])
        if self.verbose: print("DONE!")
        return answers
    
    
    def count_words(self, titles, descriptions):
        '''
        returns two sparse matrices, each for titles counts and descriptions counts
        '''
        counter = CountVectorizer(dtype=np.float, vocabulary=self.voc, token_pattern=r"(?u)\b\w+\b")
        if self.verbose: print("Counting words in titles", end='')
        tit_counts = counter.fit_transform(titles)
        if self.verbose: print(", descriptions")
        des_counts = counter.transform(descriptions)
        return tit_counts, des_counts

    
    @staticmethod
    def preprocess_polish(text):
        '''
        removes polish signs, etc.
        '''
        
        text = str(text)
        text = text.lower()
        text = re.sub('\n', ' ', text)

        text = re.sub('ą', 'a', text)
        text = re.sub('ż', 'z', text)
        text = re.sub('ź', 'z', text)
        text = re.sub('ś', 's', text)
        text = re.sub('ę', 'e', text)
        text = re.sub('ć', 'c', text)
        text = re.sub('ń', 'n', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ł', 'l', text)

        text = re.sub(' +', ' ', text)
        text = re.sub('[^a-z0-9+& ]', ' ', text)
        text = text.strip(' ')
        return text
    
    
    @staticmethod
    def choose_labels_for_text ( labels, rank, text_title, text, k=5 ):
        '''
        good heuristic to choose k labels from a bigger set of ranked labels
        '''
        
        universal_labels = ['praca', 'kurtka', 'kurtka_zimowa', 'rower', 'sukienka', 'mieszkanie', 'auto']
        stop_words = {'do', 'na', 'po', 'dla', 'z', 'od', 'w', 'a', 'r', 'p', 'o', '0'}

        if len(labels)<k:
            bag = [ x for label in labels for x in label.split('_') ]
            labels = list(Counter(labels + bag))
            labels = list(Counter(labels + universal_labels))[:k]

        assert len(labels)==len(set(labels))
        assert len(labels)>=5

        scores = np.zeros( len(labels) )
        scores[:len(rank)] = np.array(rank)
        scores[len(rank):] = 0.

        for label_num, label in enumerate(labels):
            words_list = re.sub("[^a-z0-9&+,-]", "_", label).split('_')
            words = set(words_list) - stop_words

            for word in words:
                word = word.strip('_')
                word = word.strip('&')

                if word=='': scores[label_num] *= 0
                if len(word)==1: scores [label_num] *= 0.99

                if len(word)>3:
                    if word[:-2] in text_title:
                        scores [label_num] *= 1.16
                    if word[:-1] not in text_title:
                        scores [label_num] *= 0.26
                    if word not in text_title:
                        scores [label_num] *= 0.78
                else:
                    if word in text_title:
                        scores [label_num] *= 1.21
                    if word[:-1] not in text_title:
                        scores [label_num] *= 0.44
                    if word not in text_title:
                        scores [label_num] *= 0.30
        return ' '.join(list(np.array(labels)[np.argsort( - scores, axis=0)[:5]]))
    
    
    @staticmethod
    def rate_answers(predicted_answers, true_answers):
        '''
        rates predicted answers due to the competition's formula
        '''
        
        predicted_answers = np.array([x.split(' ') for x in predicted_answers])
        calculated_max_points = np.array([0.0, 0.511111, 0.688888, 0.866666, 0.933333, 1.0])
        scores = np.array( [[ predicted_answers[i, j] in true_answers[i].split() for i in range(predicted_answers.shape[0]) ] for j in range(5)] )
        r1 = np.sum(scores[:1, :])
        r3 = np.sum(scores[:3, :]) / 3
        r5 = np.sum(scores[:5, :]) / 5
        print("Perfect order score:", np.mean(calculated_max_points[ np.sum(scores, 0) ]))
        print( "Real score:", (r1+r3+r5) / 3 / predicted_answers.shape[0] )
        return (r1+r3+r5) / 3 / predicted_answers.shape[0]