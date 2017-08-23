import argparse
import pandas
from collections import Counter, defaultdict
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    parser = argparse.ArgumentParser(description="""Convert conllu to conll format""")
    parser.add_argument('--input', help="",default='/Users/hmartine/proj/verdisandbox/data/FNpilotfirstdocs/verdipilot/a_bad_night_for_rubio.sxl.fn.pred')
    args = parser.parse_args()
    fncount=Counter()
    triggeredby=defaultdict(set)
    lmtz = WordNetLemmatizer()

    fnpanda = pandas.read_csv("../data/res/frametargetlexicon.tsv",'\t')
    onlytrain=fnpanda[fnpanda['origin'] == 'training']
    lexical_words_no_frame = Counter()
    lexpos = 'NN NNS JJ JJR JJS VB VBD VBG VBN VBP VBZ'.split()

    for line in open(args.input).readlines():
        line = line.strip()
        if line:
            line = line.split('\t')
            if line[3]:
                framename = line[3]
                fncount[framename]+=1
                triggeredby[framename].add(line[1].lower())
            else:
                if line[2] in lexpos:
                    checkpos="n"
                    if line[2].startswith("V"):
                        checkpos="v"
                    elif line[2].startswith("J"):
                        checkpos='a'
                    lemma_w = lmtz.lemmatize(line[1],pos=checkpos)
                    lexical_words_no_frame[lemma_w]+=1


#    for w in sorted(lexical_words_no_frame.keys()):
#        print(w,lexical_words_no_frame[w],w.lower() in fnpanda['lemma'].values)

    for k in sorted(fncount.keys()):
        print(k,fncount[k],sorted(triggeredby[k]), k in onlytrain['framename'].values)






if __name__ == "__main__":
    main()
