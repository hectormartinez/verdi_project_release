import argparse
import pandas as pd
import math
from collections import Counter
from nltk import agreement
import numpy as np

anns = pd.read_csv("../res/100ex_BS_HMA.csv")

BS = list(anns.BS)
HM = list(anns.HM)

annoitems = []

raw_annos = Counter()
tuples = Counter()

for c,(b,h) in enumerate(zip(BS,HM)):
    if b == "-" or h == '-':
        pass
    else:
        if math.isnan(float(b)):
            b = 0
        if math.isnan(float(h)):
            h = 0

        annoitems.append(("b",c,b))
        annoitems.append(("h",c,h))
        raw_annos[b] += 1
        raw_annos[h] += 1
        if float(b) != float(h):
            if b == 0:
                tuples["h"] += 1
            else:
                tuples["b"] += 1


task = agreement.AnnotationTask(annoitems)
print(task.alpha(),task.avg_Ao())
print(raw_annos["1"]/sum(raw_annos.values()))
print(tuples)