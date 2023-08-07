import argparse
import json
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
import torch
import torch.nn.functional as F
from kg_utils import *

parser = argparse.ArgumentParser(description="Process maxlen and archive_file.")
parser.add_argument('--archive_file', type=str, default="./checkpoint/knowbert_wiki_wordnet_model.tar.gz", help="Path to the KnowBert model archive file.")
parser.add_argument('--epoch', type=int, default=500, help="Max epoch for distillation.")

args = parser.parse_args()
maxlen = args.maxlen
epoch = args.epoch
archive_file = args.archive_file

    
# load model and batcher
params = Params({'archive_file': archive_file})
model = ModelArchiveFromParams.from_params(params=params)
model.eval()
batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')

# get bert vocab
vocab = list(batcher.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens.values())

known_entities = init_entities()
new_entities = known_entities
relations, _ = init_relations()

records = []
stopwords = ["every", "more", "it", "and", "of", "you", "he", "she", "us", "we", "him", "her", "your", "our", "ours", "yours", "no", "if", "else", "but", "though", "what", "where", "which", "why", "how", "who", "whom", "whose", "that", "all", "I", 'i', 'me', "are", "you","you're", "there", "here", "this", "that", "is","when", "very", "it","you","a", "an", "of""can", "of", "to", "a", "an", ".", ",", "[UNK]", "\"", ":",">","<","\]", "surname", "name", "term","","index","genus","category","the","these","their","those","then","article","whatever","whoever","none"] 
qry_cnt = 0
nent_cnt = 0
for epo in range(epoch):
    tmp_ent = []
    sentences, mask_ind, rel_list, ent_list = make_questions_one_word(new_entities, relations) 
    #mask_ind = [2]
    threshold = -3
    batch = 0

    for s in batcher.iter_batches(sentences, verbose=False):
        model_output = model(**s)
        # the tokenized sentence, where the 6-th token is [MASK]
        for i in range(len(model_output['contextual_embeddings'])):
            qry_cnt = qry_cnt + len(model_output['contextual_embeddings'])
            indx = batch * 8 + i
            logits, _ = model.pretraining_heads(torch.tensor([model_output['contextual_embeddings'][i].cpu().detach().numpy()]), torch.tensor([model_output['pooled_output'][i].cpu().detach().numpy()]))
            log_probs = F.log_softmax(logits, dim=-1)
            for masks in mask_ind[indx]:
                topk_vocab = torch.topk(log_probs[0, masks], 10, 0)[1]
                topk_fid = torch.topk(log_probs[0, masks], 10, 0)[0]
                for j in range(10):
                    if(topk_fid[j].item() > threshold):
                        new_rec = {"EntityA" : vocab[topk_vocab[j].item()], "EntityB" :ent_list[indx] , "Relation": rel_list[indx], "Fid": topk_fid[j].item()}
                        records.append(new_rec)
                        if(vocab[topk_vocab[j].item()] not in (list(known_entities) + list(new_entities) + list(stopwords)) and vocab[topk_vocab[j].item()] != rel_list[indx]):
                            with open("output/distilled.json",'a') as f:
                                json.dump(new_rec,f)
                                #json.dump(sentences[indx] , f)
                                f.write("\n")
                            tmp_ent.append(vocab[topk_vocab[j].item()]) 
                            nent_cnt = nent_cnt + 1     
        known_entities = list(set(new_entities).union(set(known_entities)))
        new_entities = tmp_ent
        batch = batch + 1
        
        if len(new_entities) == 0:
            break

print("Query counts:", qry_cnt)
print("New Ent counts:", nent_cnt)

