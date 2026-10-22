"""Controlled source/target transfer and retention experiment on generated text."""
from __future__ import annotations
import json, random, time
from pathlib import Path
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import vstack

SOURCE={0:["dim","weak","slow","poor","unstable"],1:["bright","strong","fast","excellent","reliable"]}
TARGET={0:["fragile","faint","sluggish","rough","inconsistent"],1:["durable","vivid","swift","premium","steady"]}
NEUTRAL=["device","system","unit","series","product","report"]
TEMPLATES=["The {noun} was {a} and {b} during evaluation.",
           "Reviewers describe the {noun} as {a}, {b}, and generally {c}.",
           "In the latest report, this {noun} performed {a}; its operation felt {b}."]

def samples(seed:int, domain:str, count:int)->tuple[list[str],list[int]]:
    rng=random.Random(seed+(0 if domain=='source' else 1000)); vocab=SOURCE if domain=='source' else TARGET
    X=[]; y=[]
    for i in range(count):
        label=i%2; words=rng.sample(vocab[label],2)+rng.sample(NEUTRAL,2)
        if rng.random()<.18: words.append(rng.choice(vocab[1-label]))
        rng.shuffle(words); template=rng.choice(TEMPLATES)
        X.append(template.format(noun=words[2],a=words[0],b=words[1],c=rng.choice(vocab[label]))); y.append(label)
    return X,y

def metrics(model, x, y)->dict[str,float]:
    pred=model.predict(x)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro'))}

def evaluate(seeds:int=3)->dict[str,Any]:
    rows=[]
    for seed in range(seeds):
        sx,sy=samples(seed,'source',160); tx,ty=samples(seed+40,'target',160)
        stx,sty=samples(seed+90,'source',80); ttx,tty=samples(seed+140,'target',80)
        # Vocabulary uses training text only; evaluation text remains unseen.
        vector=TfidfVectorizer(ngram_range=(1,2),min_df=1)
        xs=vector.fit_transform(sx+tx); xst=vector.transform(stx); xtt=vector.transform(ttx)
        source_x=xs[:len(sx)]; target_x=xs[len(sx):]
        source_model=LogisticRegression(max_iter=250,random_state=seed,solver='liblinear').fit(source_x,sy)
        t_adapter=time.perf_counter(); target_model=LogisticRegression(max_iter=250,random_state=seed,solver='liblinear').fit(target_x,ty)
        adapter_fit_s=time.perf_counter()-t_adapter
        source_terms=set(' '.join(sx).split()); target_terms=set(' '.join(tx).split())-source_terms
        feature_names=vector.get_feature_names_out(); target_mask=[1. if any(term in target_terms for term in name.split()) else 0. for name in feature_names]
        class ResidualAdapter:
            def predict(self,features):
                masked=features.multiply(target_mask)
                scores=source_model.decision_function(features)+target_model.decision_function(masked)
                return source_model.classes_[(scores>0).astype(int)]
        methods={
            'source_only':(source_x,sy),
            'target_only_adaptation':(target_x,ty),
            'source_target_replay':(vstack([source_x,target_x]), sy+ty),
        }
        for name,(train_x,train_y) in methods.items():
            t0=time.perf_counter(); model=LogisticRegression(max_iter=250,random_state=seed,solver='liblinear').fit(train_x,train_y); fit_s=time.perf_counter()-t0
            for domain,x,y in [('source',xst,sty),('target',xtt,tty)]:
                rows.append({'seed':seed,'method':name,'domain':domain,**metrics(model,x,y),
                             'fit_seconds':fit_s,'train_examples':train_x.shape[0], 'feature_count':train_x.shape[1]})
        adapter=ResidualAdapter()
        for domain,x,y in [('source',xst,sty),('target',xtt,tty)]:
            rows.append({'seed':seed,'method':'frozen_source_plus_sparse_residual','domain':domain,
                **metrics(adapter,x,y),'fit_seconds':adapter_fit_s,'train_examples':len(sy)+len(ty),
                'feature_count':int(sum(target_mask))})
    summaries=[]
    for method in sorted({r['method'] for r in rows}):
        for domain in ('source','target'):
            group=[r for r in rows if r['method']==method and r['domain']==domain]
            summaries.append({'method':method,'domain':domain,'accuracy_mean':sum(r['accuracy'] for r in group)/len(group),
                'accuracy_sd':(sum((r['accuracy']-sum(q['accuracy'] for q in group)/len(group))**2 for r in group)/max(1,len(group)-1))**.5,
                'macro_f1_mean':sum(r['macro_f1'] for r in group)/len(group), 'fit_seconds_mean':sum(r['fit_seconds'] for r in group)/len(group)})
    return {'seed_count':seeds,'dataset':'generated lexical source/target shift; disjoint deterministic splits','raw':rows,'summary':summaries}

if __name__=='__main__': print(json.dumps(evaluate(),indent=2))
