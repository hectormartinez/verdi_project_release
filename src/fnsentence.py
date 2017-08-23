import networkx as nx

class FrameEntry():
    def __init__(self, name, index, arguments=None):
        self.name = name
        self.index = index
        if not arguments:
            arguments = {}
        self.arguments = arguments

    # def __repr__(self):
    #     return str(self)

    def __str__(self):
        return self.name + "\t" + ",".join(map(str, self.arguments))


class FrameArgument():
    def __init__(self, name, start, end=None):
        self.name = name
        self.start = start
        if end:
            self.end = end
        else:
            self.end = start

    def __str__(self):
        if self.start == self.end:
            return '"' + self.name + '": "' + str(self.start + 1) + '"'
        return '"' + self.name + '": "' + str(self.start + 1) + ':' + str(self.end + 1) + '"'


class FrameSentence:
    def __init__(self, text=None, textlist=None, postags=None, heads=None, deprels=None, frames=None, preannotations=None, id_=None):
        if text is not None:
            self.text = text
            self.textlist = text.split(" ")
            self.offsetdict = {}
            self.postags = ["EMPTY"] * len(self.textlist)
            self.char_to_word_mapping = [-1] * len(self.text)

            if self.textlist[-1] == "":
                self.textlist=self.textlist[:-1]

            for i in range(len(self.textlist)):
                 if self.textlist[i] == "":
                    self.textlist[i] = "-"
            self.frames = {}
            self.buildOffsetDict()
        else:
            self.textlist = textlist
            self.forms = textlist
            self.postags = postags
            self.heads = heads
            self.deprels = deprels
            self.frames = frames
            self.preannotations = preannotations
            self.id_ = id_

            assert len(self.forms) == len(self.postags)
            assert len(self.forms) == len(self.heads)
            assert len(self.forms) == len(self.deprels)

            for frame in self.frames.values():
                assert self.preannotations[frame.index] and len(self.preannotations[frame.index]), "No preannotations for frame {} in sentence {}".format(frame.name, self.id_)


    def dep_parse(self):
        sent = nx.DiGraph()

        sent.add_node(0, {'form': 'ROOT', 'pos': 'ROOT'})
        for i in range(1, len(self.forms)):
            sent.add_node(i, {'form': self.forms[i],
                              'pos': self.postags[i]
                              })
            sent.add_edge(self.heads[i], i, deprel=self.deprels[i])

        return sent


    def buildOffsetDict(self):
        OffSetDict = {}
        prev = 0
        wordcount = 0
        char_to_word_mapping = [-1] * len(self.text)

        for i in range(len(self.text)):
            if self.text[i] == " ":
                OffSetDict[str(prev) + ":" + str(i - 1)] = wordcount
                prev = i + 1
                wordcount += 1
                char_to_word_mapping[i] = -1
            else:
                char_to_word_mapping[i] = wordcount

        OffSetDict[str(prev) + ":" + str(i - 1)] = wordcount
        self.char_to_word_mapping = char_to_word_mapping
        self.offsetdict = OffSetDict

    def addFrame(self,name,targetindex,arguments): # Obs! Used only for Gold frames,
        self.frames[targetindex]=FrameEntry(name,arguments)

    def framename_list(self):
        return [self.frames[k].name for k in self.frames.keys()]


    def heuristic_target_detection(self, k=2):
        if len(self.postags) == 0:
            raise Exception("No postags for heuristics")
        if len(self.preannotatedframes) == 0:
            raise Exception("No postags for heuristics")

        # DASETALTRIGGERS="VERB NOUN".split(" ") # Triggers according to DAS et al
        # DEVtriggers_en = "NOUN VERB ADJ PROPN DET ADV NUM ADP PRON AUX CONJ SCONJ X INT PUNCT".split(" ")
        # DEVtriggers_es = "NOUN VERB ADJ PROPN ADV NUM PRON ADP DET AUX CONJ X PUNCT SCONJ".split(" ")

        DEVtriggers_en_es = "NOUN VERB ADJ PROPN ADV DET NUM PRON ADP AUX CONJ X SCONJ PUNCT INTJ".split(" ")
        TARGETPOS = DEVtriggers_en_es[:k]

        predictedframes = {}


        for i in range(len(self.textlist)):
            if i in self.frames:
                if self.postags[i] in TARGETPOS:
                    predictedframes.predictedframes[i] = FrameEntry("FRAME",{})

        return predictedframes


    def evaluate(self,predictedframes):

        tp=0
        tn=0
        fn=0
        fp=0
        for index in range(len(self.textlist))[1:]:
            if index in self.frames.keys() and index in predictedframes.frames.keys():
                tp+=1
            if index not in self.frames.keys() and index in predictedframes.frames.keys():
                fp+=1
            if index not in self.frames.keys() and index not in predictedframes.frames.keys():
                tn+=1
            if index in self.frames.keys() and index not in predictedframes.frames.keys():
                tn+=1
        return (self.id_,tp,tn,fn,fp)


    def prettyprintframe(self, i):
        if i in self.frames:
            #for k in self.frames[i].arguments.keys():
            return self.frames[i].name+"\t{" +",".join([str(self.frames[i].arguments[k]) for k in self.frames[i].arguments.keys()])+"}"#+ ",".join([a.name+"::"+str(a.start) for a in self.frames[i].arguments]) #"\t{" + ", ".join([str(k) for k in self.frames[i].arguments]) + "}"
        else:
            return "\t{}"


    def print_predictions(self):
        acc = []
        for i in range(len(self.textlist)):
            acc.append("\t".join(
                [str(i + 1), self.textlist[i], self.postags[i], self.prettyprintframe(i)]))
        acc.append("")
        return acc


    #These two functions are only for the export frm the export_training_data, they can otherwise yield a strange offset of arguments
    def print_cst_format(self):

        acc = []
        for i in range(len(self.textlist)):
            acc.append(
                "\t".join([str(i + 1), self.textlist[i], self.postags[i], self.prettyprintframe(i)]))
        acc.append("")
        return acc

    def print_treetagger_input(self):
        acc = self.textlist + ["<SENTENCEBOUNDARY>"]
        return acc


def head_of(sent, n):
    for u, v in sent.edges():
        if v == n:
            return u
    return None
