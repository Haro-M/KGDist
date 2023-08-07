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

def make_questions_one_word(new_entities, relations):
    questions = []
    mask_indxs = []
    rel_list = []
    ent_list = []
    for entity in new_entities:
        for relation in relations:
            question1 = "The [MASK] "+relation+" "+entity+"."
            mask_ind1 = [2]
            rel1 = relation
            question2 = entity +" "+relation+" the [MASK] ."
            indx2 = len(question2.split(" "))
            mask_ind2 = [indx2-1]
            rel2 = relation
            questions.append(question1)
            print(questions)
            questions.append(question2)
            print(questions)
            mask_indxs.append(mask_ind1)
            mask_indxs.append(mask_ind2)
            rel_list.append(rel1)
            rel_list.append(rel2)
            ent_list.append(entity)
            ent_list.append(entity)
    return questions, mask_indxs, rel_list, ent_list
    
def make_questions_n_word(new_entities, relations, n):
    questions = []
    mask_indxs = []
    rel_list = []
    ent_list = []

    for entity in new_entities:
        for relation in relations:
            masks = "[MASK] " * n
            question_combinations = [
                (f"{masks}{relation} {entity}.", list(range(1,n+1)), relation, entity),
                (f"{entity} {relation} {masks}.", list(range(len(("{entity} {relation} {masks}.").split(" "))-n, len(("{entity} {relation} {masks}.").split(" ")))), relation, entity),
                (f"The {masks}{relation} {entity}.", list(range(2,2+n)), relation, entity),
                (f"{entity} {relation} the {masks}.",list(range(len(("{entity} {relation} the {masks}.").split(" "))-n, len(("{entity} {relation} the {masks}.").split(" ")))), relation, entity)
            ]

            for qc in question_combinations:
                questions.append(qc[0])
                mask_indxs.append(qc[1])
                rel_list.append(qc[2])
                ent_list.append(qc[3])

    return questions, mask_indxs, rel_list, ent_list
