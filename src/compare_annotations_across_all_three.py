import argparse
import pandas as pd
from collections import Counter
from nltk import agreement
import numpy as np

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from collections import Counter
""""HITId","HITTypeId","Title","Description","Keywords","Reward","CreationTime","MaxAssignments","RequesterAnnotation","AssignmentDurationInSeconds","AutoApprovalDelayInSeconds","Expiration","NumberOfSimilarHITs","LifetimeInSeconds","AssignmentId","WorkerId","AssignmentStatus","AcceptTime","SubmitTime","AutoApprovalTime","ApprovalTime","RejectionTime","RequesterFeedback","WorkTimeInSeconds","LifetimeApprovalRate","Last30DaysApprovalRate","Last7DaysApprovalRate","Input.row_index","Input.ref_statement","Input.ref_title","Input.ref_url","Input.target_statement","Input.target_title","Input.target_url","Input.media_source","Input.relation_type","Answer.Category","Approve","Reject"
"""

def renormalize(s):
    return s.replace("&#44;",',').replace('&#34;','"')

def simplemajority(annotations):
        threshold  = int(len(annotations))
        if annotations.count("same") > annotations.count("omission"):
            return "SAME"
        else:
            return 'OMISSION'

def threshold(annotations,theta):
    threshold = int(len(annotations))
    if annotations.count("omission") >= theta:
        return "OMISSION"
    else:
        return 'SAME'

def certainty(annotations):
    n_same = annotations.count("same")
    n_omission = annotations.count("omission")

    if n_same +1 == len(annotations) :
        return "S-TOTAL"
    elif n_omission +1 == len(annotations) :
        return "O-TOTAL"
    elif  n_same < n_omission:
        return "O-Partial"
    else:
        return "S-Partial"

def agreement(annotations):
    n_same = annotations.count("same")
    n_omission = annotations.count("omission")

    if n_same +1 == len(annotations) :
        return 1
    elif n_omission +1 == len(annotations) :
        return 1
    elif  n_same < n_omission:
        return 0.5
    else:
        return 0.5

def number(annotations):
 return sum([int(x=='omission') for x in annotations]) / len(annotations)

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--extended', default="/Users/hmartine/Dropbox/VerdiProjectFolder/binary_classifier_data_and_report/DGA_AMT_extended.csv")
    parser.add_argument('--pilot', default="/Users/hmartine/Dropbox/VerdiProjectFolder/binary_classifier_data_and_report/DGA_AMT_pilot.csv")
    parser.add_argument('--experts', default="/Users/hmartine/proj/verdisandbox/res/100ex_BS_HMA.csv")
    parser.add_argument('--adjudicated', default="/Users/hmartine/proj/verdisandbox/res/adjudicated.csv")


    args = parser.parse_args()

    PILOT = pd.read_csv(args.pilot)
    EXTENDED = pd.read_csv(args.extended)
    ADJUDICATED = pd.read_csv(args.adjudicated)
    anns = pd.read_csv(args.experts)
    BS = ["filler"]+list(anns.BS)
    HM = ["filler"]+list(anns.HM)
    adjudicated = ["filler"]+list(ADJUDICATED.ADJ)

    print("index pilot extended expert pil-ext pil-exp ext-pilot ref target".replace(" ","\t"))  # ref_statement,target_statement,simplemajority(annotations)]))

    arr_adj = []
    arr_ext = []
    arr_pil = []

    for row_index in sorted(set(list(PILOT.Input_row_index))):
        annotations_pilot = list((PILOT[PILOT.Input_row_index == row_index].Answer_Category))
        annotations_extended = list((EXTENDED[EXTENDED.Input_row_index == row_index].Answer_Category))

        ref_statement = renormalize(list(PILOT[PILOT.Input_row_index == row_index].Input_ref_statement)[0])
        target_statement = renormalize(list(PILOT[PILOT.Input_row_index == row_index].Input_target_statement)[0])
        if  row_index < 0 :
            pass
        elif row_index >= len(BS):
            pass
        elif BS[row_index] == "_" or HM[row_index] == "_" or BS[row_index] == "-" or HM[row_index] == "-":
            pass
        else:
            experts = [BS[row_index],HM[row_index]]
            n_expert = number(experts)
            n_pilot = number(annotations_pilot)
            n_extended = number(annotations_extended)
            #print("\t".join([str(x) for x in [row_index,n_pilot,n_extended,n_expert,float(n_pilot-n_extended),n_pilot-n_expert,n_extended-n_pilot,ref_statement,target_statement]]))# ref_statement,target_statement,simplemajority(annotations)]))
            #print(row_index,adjudicated[row_index])
            if adjudicated[row_index] == "-":
                pass
            else:
                arr_adj.append(float(adjudicated[row_index]))
                arr_ext.append(n_extended)
                arr_pil.append(n_pilot)
    arr_adj = np.array(arr_adj)
    arr_ext = np.array(arr_ext)
    arr_pil = np.array(arr_pil)

    agr_ext = [1 if x > .79 or x < 0.21 else 0 for x in arr_ext ]
    agr_pil = [1 if x > .79 or x < 0.21 else 0 for x in arr_pil ]

    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import pearsonr
    from collections import Counter
    print(cosine_similarity(arr_adj,arr_ext))
    print(cosine_similarity(arr_adj,arr_pil))
    print(cosine_similarity(arr_ext,arr_pil))

    print(cosine_similarity(agr_ext,agr_pil))



    print(arr_adj)
if __name__ == "__main__":
    main()