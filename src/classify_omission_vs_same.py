import argparse
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.preprocessing import Normalizer
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords

from sklearn.dummy import DummyClassifier
import itertools

RANDOMSTATE=112

trainingbow = set()

ner_path_bin = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/stanford-ner.jar"
ner_path_model = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.conll.4class.distsim.crf.ser.gz"
#ner_path_model = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz"

from nltk.tag.stanford import StanfordNERTagger

class StatementPair:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ?".split()
        return sl

    def __init__(self,row_index,ref_statement,target_statement,annotation,ner_tagger=None):
        self.ref_statement = ref_statement #sentence is a list of forms
        self.target_statement = target_statement
        self.row_index = int(row_index)
        self.label = 0 if annotation == "SAME" else 1
        if ner_tagger:
            self.ref_ner = ner_tagger.tag(self.ref_statement)
            self.target_ner = ner_tagger.tag(self.target_statement)
        else:
            self.ref_ner =  [(x,"X") for x in self.ref_statement if x[0].isupper()]
            self.target_ner = [(x,"X") for x in self.target_statement if x[0].isupper()]



    def _word_venn_diagram(self):
        commonwords = set(self.ref_statement).intersection(set(self.target_statement))
        onlyref = set(self.ref_statement).difference(set(self.target_statement))
        onlytarget =  set(self.target_statement).difference(set(self.ref_statement))
        return  commonwords,onlyref,onlytarget

    def _get_average_vector(self,wordset,embeddings):
        wordvecs = [embeddings["DEFAULT"]]
        for word in wordset:
            if word in embeddings:
                 wordvecs.append(embeddings[word])

        wordvecs = np.array(wordvecs)
        wv =  wordvecs.sum(axis=0) / (len(wordvecs))
        return wv

    def a_dicecoeff(self):
        D = {}
        try:
            commonwords, onlyref, onlytarget = self._word_venn_diagram()

            D["a_dicecoeff"] = len(commonwords) / len(self.ref_statement)
            D["a_onlyref"] = len(onlyref) / len(commonwords)
        except:
            D["a_dicecoeff"] = 0
            D["a_onlyref"] = 0
        return D

    def b_lengths(self):
        D = {}
        D["b_lenref"] = len(self.ref_statement)
        D["b_lentarget"] = len(self.target_statement)
        D["b_lendiff"] = len(self.ref_statement)- len(self.target_statement)
        return D

    def c_bow(self,bowfilter=None):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()
        if bowfilter:
            for b in onlyref:
                if b in bowfilter:
                    D["c_r_"+b]=1
        else:
            for b in onlyref:
                D["c_r_" + b] = 1
        return D

    def d_embeds(self,embeddings):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()

        common_average_vector = self._get_average_vector(commonwords,embeddings) #THIS DOES NOT HELP
        #for i,v in enumerate(common_average_vector):
        #    D["c_com_"+str(i)]=v
        ref_average_vector = self._get_average_vector(onlyref, embeddings)
        for i, v in enumerate(ref_average_vector):
            D["d_ref_" + str(i)] = v
        #tgt_average_vector = self._get_average_vector(onlytarget, embeddings)
        #for i, v in enumerate(tgt_average_vector):
        #    D["c_tgt_" + str(i)] = v
        #co = cosine(ref_average_vector,tgt_average_vector)
        #D["c_dif"] = 0 if np.isnan(co) else co
        return D

    def e_stop(self):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()
        D["e_prop_stop"] = len([x for x in onlyref if x in self.stoplist()])/len(onlyref) if len(onlyref) > 0 else 0
        return D

    def _ner_sequences(self,taggedarray):
        #TODO try all models (3/4/7) and see if it makes a difference, which it does not.
        acc = []
        sequences = set()
        for w, t in taggedarray:
            if t != "O":
                acc.append(w)
            elif acc:
                sequences.add(" ".join(acc))
                acc = []
        if acc:
            sequences.add(" ".join(acc))
            acc = []
        return sequences



    def f_ner(self):
        D = {}
        ref_seqs=self._ner_sequences(self.ref_ner)
        target_seqs=self._ner_sequences(self.target_ner)
        #print(ref_seqs.difference(target_seqs))
        D["f_ner"] = len(ref_seqs.difference(target_seqs))
        return D


    def featurize(self,variant,embeddings,bowfilter=None):
        D = {}
        if "a" in variant:
            D.update(self.a_dicecoeff())
        if "b" in variant:
            D.update(self.b_lengths())
        if "c" in variant:
            D.update(self.c_bow(bowfilter))
        if "d" in variant:
            D.update(self.d_embeds(embeddings))
        if "e" in variant:
            D.update(self.e_stop())
        if "f" in variant:
            D.update(self.f_ner())
        return D



def getStatementPairs(infile,ner_tagger):
    pairs = []
    for line in open(infile).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement = wordpunct_tokenize(ref_statement)
        target_statement = wordpunct_tokenize(target_statement)
        sp = StatementPair(row_index, ref_statement, target_statement, annotation,ner_tagger)
        pairs.append(sp)
    return pairs




def collect_features(pairs,variant,embeddings):
    labels = [sp.label for sp in pairs]
    featuredicts = [sp.featurize(variant,embeddings) for sp in pairs]

    vec = DictVectorizer()
    norm = Normalizer()
    features = vec.fit_transform(featuredicts)#.toarray()
    labels = np.array(labels)
    return features, labels, vec



def crossval(features, labels,variant,printcoeffs=False):
    maxent = LogisticRegression(penalty='l2')
    dummyclass = DummyClassifier("most_frequent")
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too


    scores = defaultdict(list)
    TotalCoeffCounter = Counter()
    allpreds =[]
    alldummy = []
    allgold = []
    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=RANDOMSTATE):
        #print(TestIndices)
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels[TestIndices]
        dummyclass.fit(TrainX_i,Trainy_i)
        maxent.fit(TrainX_i,Trainy_i)

        ypred_i = maxent.predict(TestX_i)
        ydummypred_i = dummyclass.predict(TestX_i)
        #coeffs_i = list(maxent.coef_[0])
        #coeffcounter_i = Counter(vec.feature_names_)
        #for value,name in zip(coeffs_i,vec.feature_names_):
        #    coeffcounter_i[name] = value
        allpreds.extend(ypred_i)
        allgold.extend(Testy_i)
        alldummy.extend(ydummypred_i)

        acc = accuracy_score(ypred_i, Testy_i)
        #pre = precision_score(ypred_i, Testy_i,pos_label=1)
        #rec = recall_score(ypred_i, Testy_i,pos_label=1)
        f1 = f1_score(ypred_i, Testy_i,pos_label=1)

        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        #scores["Precision"].append(pre)
        #scores["Recall"].append(rec)

        #
        acc = accuracy_score(ydummypred_i, Testy_i)
        # pre = precision_score(ydummypred_i, Testy_i,pos_label=1)
        # rec = recall_score(ydummypred_i, Testy_i,pos_label=1)
        f1 = f1_score(ydummypred_i, Testy_i,pos_label=1)
        #
        scores["dummy-Accuracy"].append(acc)
        scores["dummy-F1"].append(f1)
        # scores["dummy-Precision"].append(pre)
        # scores["dummy-Recall"].append(rec)

        #posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        #negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))

    #print("Pervasive positive: ", posfeats)
    #print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    #print("--")
    #for key in sorted(scores.keys()):
    #    currentmetric = np.array(scores[key])
        #print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
        #print("%s : %0.2f" % (key,currentmetric.mean()))
    print("%s %.3f %.3f" % (variant,np.array(scores["dummy-Accuracy"]).mean(),np.array(scores["dummy-F1"]).mean()),end="")

    fout = open(variant+".labels",mode="w")
    fout.write(" ".join([str(s) for s in allpreds]))
    fout.close()

    fout = open("dummy.labels", mode="w")
    fout.write(" ".join([str(s) for s in alldummy]))
    fout.close()

    fout = open("gold.labels", mode="w")
    fout.write(" ".join([str(s) for s in allgold]))
    fout.close()

    if printcoeffs:

        maxent.fit(features,labels) # fit on everything

        coeffs_total = list(maxent.coef_[0])
        for (key,value) in TotalCoeffCounter.most_common()[:20]:
            print(key,value)
        print("---")
        for (key,value) in TotalCoeffCounter.most_common()[-20:]:
            print(key,value)

def load_embeddings(embedpath):
    E = {}
    for line in open(embedpath).readlines():
        a=line.split()
        E[a[0]]=np.array([float(x) for x in a[1:]])
    E["DEFAULT"]=(np.array([0]*len(E[list(E.keys())[0]]))) # zero times the size of the first embedding
    return E


def splits_on_expert_fold(adjudicatedpath,pairs):
    adjudicated = list(pd.read_csv(adjudicatedpath)["ADJ"])
    filteredpairs = []
    test_section = []
    train_section = []
    expert_y = [int(x) for x in adjudicated if x != '-'][:100]
    for i, sp in enumerate(pairs):
        if i < len(adjudicated) and adjudicated[i] == '-' and len(test_section) < 100:
            pass
        elif i < len(adjudicated) and (adjudicated[i] == '0' or adjudicated[i] == '1') and len(test_section) < 100:
            test_section.append(sp)
        elif i >= len(adjudicated):
            train_section.append(sp)
        else:
            # These are the few examples that go into training because they are over  100
            train_section.append(sp)
    return train_section,test_section,expert_y


def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_extendedamt_simplemajority.tsv")
    parser.add_argument('--embeddings', default="/Users/hmartine/data/glove.6B/glove.6B.50d.txt")
    parser.add_argument('--adjudicated', default="../res/adjudicated.csv")
    parser.add_argument('--ner_path_model', default="/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.conll.4class.distsim.crf.ser.gz")

    args = parser.parse_args()

    E = load_embeddings(args.embeddings)
    ner_tagger =  StanfordNERTagger(args.ner_path_model,ner_path_bin)

    pairs = getStatementPairs(args.input,ner_tagger)
    train_section, test_section, expert_test_y=splits_on_expert_fold(args.adjudicated,pairs)

    #print("reading and NER done")
    letter_ids = "abcdef"
    variants = []
    for k in range(1,7):
        variants.extend(["".join(x) for x in itertools.combinations(letter_ids,k)])



    for variant in variants:# ["a","b","c","d","e","f","ab","ac","ad","ae","af","bc","bd","be","bf","cd","ce","cf","abc","cde","abd","abf","bcf","cef","abde","abdf","abcde","abcdef"]:

        features, labels, vec = collect_features(pairs,variant,embeddings=E)
        crossval(features, labels,variant)
        all_feats, all_labels, vec = collect_features(test_section+train_section,variant,E)

        test_X = all_feats[:len(expert_test_y)]
        train_X = all_feats[len(expert_test_y):]
        turker_test_y = all_labels[:len(expert_test_y)]
        turker_train_y =  all_labels[len(expert_test_y):]
        maxent = LogisticRegression(penalty='l2')
        #maxent = DummyClassifier("most_frequent")
        maxent.fit(train_X,turker_train_y)
        y_pred=maxent.predict(test_X)

        acc_turker = accuracy_score(y_pred, turker_test_y)
        f1_turker = f1_score(y_pred, turker_test_y, pos_label=1)
        acc_expert = accuracy_score(y_pred, expert_test_y)
        f1_expert = f1_score(y_pred, expert_test_y, pos_label=1)
        print(" %.3f %.3f %.3f %.3f" % (acc_turker,f1_turker,acc_expert,f1_expert))


if __name__ == "__main__":
    main()
