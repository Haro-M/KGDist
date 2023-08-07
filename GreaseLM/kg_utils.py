import sys

experiment_data = 'squad'
experiment_model = 'Bert'
use_spacy = True
if len(sys.argv) > 1:
    experiment_data = str(sys.argv[1])
    experiment_model = str(sys.argv[2])
    use_spacy = True if len(sys.argv) >= 4 else False

import urllib.request
import zipfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import spacy
import textacy
import en_core_web_sm
from spacy.symbols import ORTH

global entities_list_global
global relation_list_global
entities_list_global = []
relation_list_global = []

def update_spacy_dict(nlp, entities_list):
    for ent in (entities_list):
        special_case = [{ORTH: ent}]
        nlp.tokenizer.add_special_case(ent, special_case)
    
def textacy_extract_relations(text, entities_list):
    nlp = en_core_web_sm.load()
    update_spacy_dict(nlp, entities_list)
    doc = nlp(text)
    return textacy.extract.subject_verb_object_triples(doc)

def spacy_extract_relations(text, entities_list):
    nlp = en_core_web_sm.load()
    update_spacy_dict(nlp, entities_list)
    doc = nlp(text)
    triples = []

    for ent in doc.ents:
        preps = [prep for prep in ent.root.head.children if prep.dep_ == "prep"]
        for prep in preps:
            for child in prep.children:
                triples.append((ent.text, "{} {}".format(ent.root.head, prep), child.text))
    return triples
    
def customize_extract_relations(text, relation_list):
    relation_list = relation_list
    triples = []
    for i in relation_list:
        if(text.find(i) > 0):
            EntityA = text[0:text.find(i)-1].title()
            EntityB = text[text.find(i)+len(i)+1:].title()
            triples.append((EntityA, i, EntityB))
    return triples

def generate_kg(predicted_sentences, fidelities, use_spacy, relation_list):
    label_dict = {}
    fid_dict = {}
    row_list = []
    next_epoch_entities = []
    '''
    if use_spacy:
        extract_relations = spacy_extract_relations
    else:
        extract_relations = textacy_extract_relations
        '''
    extract_relations = customize_extract_relations
        
    cnt = 0
    for text in predicted_sentences:
        relations = extract_relations(text, relation_list)
        fid = fidelities[cnt]
        for _source, _relation, _target in relations:
          _source = _source.title()
          _target = _target.title()
          if(_source != _target):
              if(label_dict.get((str(_source), str(_target)))== None):
                  if(label_dict.get((str(_target), str(_source))) != None):
                      if(fid_dict[(str(_target), str(_source))] < fid.item()):
                          label_dict[(str(_source), str(_target))] = str(_relation)
                          fid_dict[(str(_source), str(_target))] = fid.item()
                          del label_dict[str(_target), str(_source)]
                          del fid_dict[str(_target), str(_source)]
                  else:    
                      label_dict[(str(_source), str(_target))] = str(_relation)
                      fid_dict[(str(_source), str(_target))] = fid.item()
              else:
                  if(fid_dict[(str(_source), str(_target))] < fid.item()):
                      label_dict[(str(_source), str(_target))] = str(_relation)
                      fid_dict[(str(_source), str(_target))] = fid.item()
        cnt = cnt + 1
    for item in label_dict.items():
        (key1, key2) = item[0]
        value = item[1]
        row_list.append({'source': key1, 'target':key2, 'edge': value, 'fid_score': str(fid_dict[(key1, key2)])})
        next_epoch_entities.append((fid_dict[(key1, key2)], key1, key2))
    return pd.DataFrame(row_list), label_dict, next_epoch_entities

def plot_kg(df, label_dict, node_color='skyblue', font_color='red', save_name='img.jpg'):
    G=nx.from_pandas_edgelist(df, "source", "target",
                              edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12,12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color=node_color, edge_cmap=plt.cm.Blues, pos = pos)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=label_dict,font_color=font_color)
    plt.savefig(save_name)

def kg_prefine(nesting_relation_list, nesting_relation_depth_limit):
    if(nesting_relation_depth_limit == -1): 
        return
    kg = pd.read_csv('test_kg_df.csv')
    kg = kg.sort_values(by=['source', 'edge', 'fid_score'], ascending=[False, False, False])
    kg_ = pd.DataFrame()
    attr_buffer = ('XXX','XXX')
    cnt = 0
    CNT0 = 0
    attr_counter = 0
    for index, row in kg.iterrows():
        CNT0 = CNT0 + 1
        source = getattr(row,'source')
        edge = getattr(row,'edge')
        if((source, edge) == attr_buffer):
            attr_counter = attr_counter + 1
        else:
            attr_counter = 0
            attr_buffer = (source, edge)
        if(attr_counter < nesting_relation_depth_limit):
            kg_ = pd.concat([kg_, row])
            cnt = cnt + 1
    print(cnt, CNT0)
    
    return

def distill_test(predicted_sentences, fidelities, use_spacy, focus_entities, relation_list, epoch, chunk_query_flag = 0, fid_threshold=20):

    kg_df, label_dict, distilled_relations = generate_kg(predicted_sentences, fidelities, use_spacy, relation_list)
    
    import os
    
    file_path = 'test_kg_df.csv'
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            content = file.read().strip() 
            if content:
                df0 = pd.read_csv(file_path)
                df0 = pd.concat([df0, kg_df])
                df0.to_csv(file_path, index=None)
            else:
                kg_df.to_csv(file_path, index=None)
    else: 
        kg_df.to_csv(file_path, index=None)   
        
    next_epo_entities = []
    distilled_relations.sort()
    for item in distilled_relations:
        (score, ent1, ent2) = item
        ent1 = ent1.lower().replace("_", " ")
        ent2 = ent2.lower().replace("_", " ")
        if score > fid_threshold:
            if ent1 in focus_entities and ent2 not in focus_entities:
                next_epo_entities.append(ent2)
            elif ent2 in focus_entities and ent1 not in focus_entities:
                next_epo_entities.append(ent1)
   
    '''
    plot_kg(kg_df, label_dict, 'skyblue', 'red', \
            'test_kg.jpg')
            '''
    print('Done!')
    return next_epo_entities


def write_questions(qid, question_concept, stem, choices, question_path):
    choices_body = []
    ch_cnt = 0
    for choice in choices:
        choice_box = {"label": chr(ord('A') + ch_cnt), "text": choice}
        choices_body.append(choice_box)
        ch_cnt = ch_cnt + 1
    question_body = {"id": str(qid), "question": {"question_concept": question_concept, "choices": choices_body, "stem": stem}}
    
    with open(question_path,'a') as f:
        json.dump(question_body,f)
        f.write("\n")


def init_entities():
    entities_list = []
    entities_path = "./distill_lab/entities.txt"
        
    f = open(entities_path, "r",encoding='utf-8')
    line = f.readline()
    while line:
        txt_data = line.replace('\n','')
        entities_list.append(txt_data)
        line = f.readline()
    
    return entities_list
        
def init_relations():
    relation_list = []
    nesting_relation_list = []
    relation_path = "./distill_lab/relations.txt"
    
        
    f = open(relation_path, "r",encoding='utf-8')
    line = f.readline()
    while line:
        txt_data = line.replace('\n','').split(',')[0]
        relation_list.append(txt_data)
        if(line.replace('\n','').split(',')[1][1] == '1'):
            nesting_relation_list.append(txt_data)
        line = f.readline()
        relation_list.append('is related to')
    return relation_list, nesting_relation_list
        
def make_questions(entities_list, unsearched_list, relation_list, query_batchsize = 5, chunk_query_flag = 0):

    question_path = './data/distill_obqa/distill_rand_split_no_answers.jsonl'
    #clear a json
    
    open(question_path,'w').close()
        
    qid = 0
    et_cnt = len(entities_list)
    e_cnt = len(unsearched_list)
    r_cnt = len(relation_list)
    
    entities_list_global = entities_list + unsearched_list
    relation_list_global = relation_list
    
    for i in entities_list:
        j = 0
        while j < e_cnt:
            question_concept = i
            choices = []
                
            for ch in range(query_batchsize):
            
                if (j + ch >= e_cnt):
                    choices.append(unsearched_list[ch % e_cnt].replace('_',' '))
                    continue

                    
                choices.append(unsearched_list[(j+ch) % e_cnt].replace('_',' '))
                j = j + 1
                
            if(len(choices) == query_batchsize - 1):   
                choices.append(unsearched_list[0])     
                
            if(len(choices) != 0):        
                for r in relation_list:
                    stem = 'Which ' + r +' ' + i + '?'
                    write_questions(qid, question_concept, stem, choices, question_path)
                    qid = qid + 1
                    if(chunk_query_flag == 0):
                        stem = i + ' '+ r +' which' + '?'
                        write_questions(qid, question_concept, stem, choices, question_path)
                        qid = qid + 1
             
    for i in range(et_cnt):
        j = i + 1
        while j < et_cnt - i:
            question_concept = entities_list[i]
            choices = []
            for ch in range(query_batchsize):
                if (j + ch >= et_cnt - i):
                    choices.append(entities_list[ch % et_cnt].replace('_',' '))
                    continue
                    
                choices.append(entities_list[(j+ch) % et_cnt].replace('_',' '))
                j = j + 1
                
            if(len(choices) == query_batchsize - 1):   
                choices.append(entities_list[0])     
                
            if(len(choices) != 0):        
                for r in relation_list:
                    stem = 'Which ' + r +' ' + entities_list[i] + '?'
                    write_questions(qid, question_concept, stem, choices, question_path)
                    qid = qid + 1
                    stem = entities_list[i] + ' '+ r +' which' + '?'
                    write_questions(qid, question_concept, stem, choices, question_path)
                    qid = qid + 1
                  
    return qid