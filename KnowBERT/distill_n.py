import json
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
import torch
import torch.nn.functional as F
from kg_utils import *
import argparse

import string

parser = argparse.ArgumentParser(description="Process maxlen and archive_file.")
parser.add_argument('--maxlen', type=int, default=3, help="Max length for sentences.")
parser.add_argument('--epoch', type=int, default=500, help="Max epoch for distillation.")
parser.add_argument('--archive_file', type=str, default="./checkpoint/knowbert_wiki_wordnet_model.tar.gz", help="Path to the KnowBert model archive file.")

args = parser.parse_args()
epoch = args.epoch
maxlen = args.maxlen
archive_file = args.archive_file

def is_only_punctuation(s):
    return all(char in string.punctuation for char in s)

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
cache = {}

def is_adverb(word):
    if word in cache:
        return cache[word]
    synsets = wordnet.synsets(word, pos=wordnet.ADV)
    result = len(synsets) > 0
    cache[word] = result
    return result


# load model and batcher
params = Params({'archive_file': archive_file})
model = ModelArchiveFromParams.from_params(params=params)
model.eval()
batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')

def has_adjacent_words(phrase):
    words = phrase.split()
    return any(words[i] == words[i + 1] for i in range(len(words) - 1))

cache = {}

# get bert vocab
vocab = list(batcher.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens.values())

known_entities = init_entities()
relations, _ = init_relations()

stopwords = ["every", "more", "it", "and", "of", "you", "he", "she", "us", "we", "him", "her", "your", "our", "ours", "yours", "no", "if", "else", "but", "though", "what", "where", "which", "why", "how", "who", "whom", "whose", "that", "all", "I", 'i', 'me', "are", "you","you're", "there", "here", "this", "that", "is","when", "very", "its","you","a", "an", "of""can", "of", "to", "a", "an", ".", ",", "[UNK]", "\"", ":",">","<","\]", "surname", "name", "term","","index","genus","category","the","these","their","those","then","article","whatever","whoever","none", "list", "include", "most", "many", "some", "everyone", "nor", "other", "class", "another", "each", "one", "resulting", "remaining", "such", "entire","was","any","so","only","also","has","its","his","neither","both","*", "be", "maybe", "my", "...", "&", "they", "influenced", "almost", "sometimes", "usually", "her", "either", "eventually","does", "yourself", "it", "di", "ot", "them","my","since","can","also","however","affects","is","lies","belonged","belonging","caused","onto","if","included","certainly","influencing","were","causing","being","aforementioned","several","similar","influences", "belongs","influences","causes","cause","influence","belong","affect","affects","effects","produces","from",",","lower","official","smallest","typical","total"] 
qcnt = 0
necnt = 0
qcnt_filtered = 0
necnt_filtered = 0

def split_array_into_n_sections(range_, n):
    section_size = range_ // n
    remainder = range_ % n
    
    result = []
    head = 0

    for i in range(n):
        tail = head + section_size
        if i < remainder:
            tail += 1
        result.append([head, tail])
        head = tail

    return result


for epo in range(epoch):
    tmp_ent = []
    for n in range(2, maxlen):
        new_entities = known_entities    
        sentences, mask_ind, rel_list, ent_list = make_questions_n_word(new_entities, relations, n) 
        threshold = -999
        indx = 0
        indexes = split_array_into_n_sections(len(sentences), 100)
        for [head, tail] in indexes:
            for s in batcher.iter_batches(sentences[head:tail], verbose=False):
                model_output = model(**s)
                for i in range(len(model_output['contextual_embeddings'])):
                    qcnt = qcnt + 1
                    logits, _ = model.pretraining_heads(torch.tensor([model_output['contextual_embeddings'][i].cpu().detach().numpy()]), torch.tensor([model_output['pooled_output'][i].cpu().detach().numpy()]))
                    log_probs = F.log_softmax(logits, dim=-1)
                    #print(mask_ind[indx])
                    answers = []
                    for masks in mask_ind[indx]:
                        tmp = []
                        topk_vocab = torch.topk(log_probs[0, masks], 10, 0)[1]
                        topk_fid = torch.topk(log_probs[0, masks], 10, 0)[0]
                        for j in range(10):
                            if(topk_fid[j].item() > threshold):
                                tmp.append((vocab[topk_vocab[j].item()], topk_fid[j].item()))
                        answers.append(tmp)
                        
                    entity_multi = ""
                    entity_fid = 0
                    signal = 1
                    for nn in range(n):
                        if len(answers[nn]) > 0:
                            for nnn in range(len(answers[nn])):
                                (ent, fid) = answers[nn][nnn]
                                
                                if ent not in rel_list[indx] and ent not in ent_list[indx] and "#" not in ent and is_adverb(ent) == False and is_only_punctuation(ent) == False and len(ent)>1:
                                    signal = 0
                                    break
                            if signal == 0:
                                entity_multi = entity_multi + ent + " "
                                entity_fid = entity_fid + fid
                            else:
                                break
                            
                    entity_multi = entity_multi.rstrip(" ")
                    
                    
                    if(entity_multi not in (list(known_entities) + list(new_entities) + list(stopwords)) and entity_multi != rel_list[indx] and signal == 0 and has_adjacent_words(entity_multi) == False and "#" not in entity_multi and entity_fid >= (-3.5*n)):
                    
                        new_rec = {"EntityA" :entity_multi, "EntityB" :ent_list[indx] , "Relation": rel_list[indx], "Fid": entity_fid}
                        
                        with open("./distilled.json",'a') as f:
                            json.dump(new_rec,f)
                            json.dump(sentences[indx] , f)
                            f.write("\n")
                        tmp_ent.append(entity_multi)
                        label = 1
                        for word in stopwords:
                            if word in entity_multi:
                                label = 0       
                        necnt = necnt + label
                    known_entities = list(set(new_entities).union(set(known_entities)))
                    new_entities = list(set(tmp_ent))
                    indx = indx + 1
                        
            if len(new_entities) == 0:
                break

for ent in known_entities:
    with open("./output/entities_distilled.json",'a') as f:
        json.dump(ent,f)
        f.write("\n")

print("Query counts:", qcnt)
print("New Ent counts:", necnt)