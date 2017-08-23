import argparse
import pandas as pd
from collections import Counter
from nltk import agreement
import numpy as np
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

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="/Users/hmartine/Dropbox/VerdiProjectFolder/binary_classifier_data_and_report/DGA_AMT_extended.csv")
    parser.add_argument('--mode', choices=['MACE', 'text', 'agreement','quantified'],default='text')
    parser.add_argument('--max', default=10000,type=int)


    args = parser.parse_args()

    DGA = pd.read_csv(args.input)

    if args.mode == "MACE":
        turkerlist =  sorted(set(list(DGA.WorkerId)))
        for row_index in sorted(set(list(DGA.Input_row_index))):
            blankline = [""]*len(turkerlist)
            annotations = list((DGA[DGA.Input_row_index == row_index].Answer_Category))
            turkers =  list((DGA[DGA.Input_row_index == row_index].WorkerId))
            for a,t in zip(annotations,turkers):
                blankline[turkerlist.index(t)]=a
            print(",".join(blankline))
    elif args.mode == 'agreement':
        certCount = Counter()
        raw_annos = Counter()
        for row_index in sorted(set(list(DGA.Input_row_index))):
            if row_index > 0 and row_index  < args.max:
                annotations = list((DGA[DGA.Input_row_index == row_index].Answer_Category))
                certCount[certainty(annotations)]+=1
                for a in annotations:
                    raw_annos[a]+=1
        print(certCount)
        print("Ommi",raw_annos["omission"]/(sum(raw_annos.values())))
        anno_items = []
        T = []
        for row_index in sorted(set(list(DGA.Input_row_index))):
            if row_index > 0 and row_index  < args.max:
                annotations = list((DGA[DGA.Input_row_index == row_index].Answer_Category))
                turkers = list(DGA[DGA.Input_row_index == row_index].WorkerId)
                T.extend(list(DGA[DGA.Input_row_index == row_index].WorkTimeInSeconds.values))

                for a, t in zip(annotations, turkers):
                    anno_items.append((t,row_index,a)) #coder,item,label
        task = agreement.AnnotationTask(anno_items)
        T = np.array(T)
        print(T.mean(),np.median(T), task.avg_Ao(),task.alpha(),task.Do_alpha(),task.multi_kappa())
    elif args.mode == 'text':
        for row_index in sorted(set(list(DGA.Input_row_index))):
            annotations = list((DGA[DGA.Input_row_index == row_index].Answer_Category))
            ref_statement = renormalize(list(DGA[DGA.Input_row_index == row_index].Input_ref_statement)[0])
            target_statement = renormalize(list(DGA[DGA.Input_row_index == row_index].Input_target_statement)[0])
            if  row_index < 0:
                pass
            else:
                print("\t".join([str(row_index),ref_statement,target_statement,simplemajority(annotations)]))
    elif args.mode == 'quantified':
        Q = []
        for row_index in sorted(set(list(DGA.Input_row_index))):
            annotations = list((DGA[DGA.Input_row_index == row_index].Answer_Category))
            q = [ int(a == 'omission') for a in annotations]
            Q.append(str(sum(q)/(len(q))))
        print(args.input,",".join(Q))

if __name__ == "__main__":
    main()