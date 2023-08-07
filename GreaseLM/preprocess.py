import os
import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.convert_obqa import convert_to_obqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts__use_LM
import pandas as pd

input_paths = {
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/dev_rand_split.jsonl',
    },
    'distill_csqa': {
        'distill': './data/distill_csqa/distill_rand_split_no_answers.jsonl',
    },
    'distill_obqa': {
        'distill': './data/distill_obqa/distill_rand_split_no_answers.jsonl',
    },
    'distill_csqa_eval': {
        'distill': './data/distill_qa_eval/distill_rand_split_no_answers.jsonl',
    },
    'distill_obqa_eval': {
        'distill': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/output.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'distill_cpnet': {
        'csv': './data/distill_cpnet/conceptnet.en.csv',
        'vocab': './data/distill_cpnet/concept.txt',
        'patterns': './data/distill_cpnet/matcher_patterns.json',
        'unpruned-graph': './data/distill_cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/distill_cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/dev.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/dev.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/dev.graph.adj.pk',
        },
    },
    'distill_csqa': {
        'statement': {
            'distill': './data/distill_csqa/statement/distill.statement.jsonl',
        },
        'grounded': {
            'distill': './data/distill_csqa/grounded/distill.grounded.jsonl',
        },
        'graph': {
            'adj-distill': './data/distill_csqa/graph/distill.graph.adj.pk',
        },
    },
    'distill_csqa_eval': {
        'statement': {
            'distill': './data/distill_csqa_eval/statement/distill.statement.jsonl',
        },
        'grounded': {
            'distill': './data/distill_csqa_eval/grounded/distill.grounded.jsonl',
        },
        'graph': {
            'adj-distill': './data/distill_csqa_eval/graph/distill.graph.adj.pk',
        },
        'statement_comparison': {
            'distill': './data/distill_csqa_eval/statement_comp/distill.statement.jsonl',
        },
        'grounded_comparison': {
            'distill': './data/distill_csqa_eval/grounded_comp/distill.grounded.jsonl',
        },
        'graph_comparison': {
            'adj-distill': './data/distill_csqa_eval/graph_comp/distill.graph.adj.pk',
        },
    },'distill_obqa_eval': {
        'statement': {
            'distill': './data/distill_obqa_eval/statement/distill.statement.jsonl',
        },
        'grounded': {
            'distill': './data/distill_obqa_eval/grounded/distill.grounded.jsonl',
        },
        'graph': {
            'adj-distill': './data/distill_obqa_eval/graph/distill.graph.adj.pk',
        },
        'statement_comparison': {
            'distill': './data/distill_obqa_eval/statement_comp/distill.statement.jsonl',
        },
        'grounded_comparison': {
            'distill': './data/distill_obqa_eval/grounded_comp/distill.grounded.jsonl',
        },
        'graph_comparison': {
            'adj-distill': './data/distill_obqa_eval/graph_comp/distill.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
    'distill_obqa': {
        'statement': {
            'distill': './data/distill_obqa/statement/distill.statement.jsonl',
        },
        'grounded': {
            'distill': './data/distill_obqa/grounded/distill.grounded.jsonl',
        },
        'graph': {
            'adj-distill': './data/distill_obqa/graph/distill.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': './data/obqa/statement/train-fact.statement.jsonl',
            'dev': './data/obqa/statement/dev-fact.statement.jsonl',
            'test': './data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test-fact.jsonl',
        },
    },
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))



def preprocess_eval():
    nprocs = 6
    routines = {
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], nprocs)}]
    }
    
    for rt in ['csqa']:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run.')
    


    
def relation_shift(relation):
    equal_PartOf = ['locates in']
    if relation in equal_PartOf:
        return ['partof', 'relatedto']
    else:
        return['relatedto']


def fid_score_shift(fid, max_fid_lim, relatedto = 0):
    upper_bound_fid = 30
    lower_bound_fid = 15 
    if relatedto != 1:
        if(fid > upper_bound_fid):
            fid = upper_bound_fid
        fid_ = (fid - lower_bound_fid) / (upper_bound_fid - lower_bound_fid) * max_fid_lim
    else:
        if fid < 20:
            fid_ = 0.5
        else:
            fid_ = 1.0
    return fid_
    
def reformat_kg(edge_path, entity_path):
    #Write entities.
    entities_list = []
    entities_path = "/home/lpz/MHL/KG_Distillation/qagnn/entities_in_question/entities.txt"
        
    f = open(entities_path, "r",encoding='utf-8')
    line = f.readline()
    while line:
        txt_data = line.replace('\n','').replace(' ','_').lower()
        entities_list.append(txt_data)
        line = f.readline()
        
    f = open(entity_path, 'w')
    for ent in entities_list:
        f.write(ent+"\n")
    f.close()
    
    #Write edges.
    kg = pd.read_csv("/home/lpz/MHL/KG_Distillation/GreaseLM/test_kg_df_obqa.csv")
    kg_ = pd.DataFrame(columns=['rel', 'sor', 'tar', 'fid'])
    for index, row in kg.iterrows():
        source = getattr(row,'source').replace(' ','_').lower()
        edge = relation_shift(getattr(row,'edge'))
        target = getattr(row, 'target').replace(' ','_').lower()
        fid = float(getattr(row, 'fid_score'))
        for e in edge:
            if(e != 'relatedto'):
                kg_ = kg_.append({'rel': e, 'sor':source, 'tar': target, 'fid':str(fid_score_shift(fid, 2, 0))}, ignore_index=True) 
            else:
                kg_ = kg_.append({'rel': e, 'sor':source, 'tar': target, 'fid':str(fid_score_shift(fid, 1, 1))}, ignore_index=True)
                
    kg_.to_csv(edge_path, sep='\t', index=None, header=None)
        
    
def distill_eval_preprocess_func():

    nprocs = 6
    
    reformat_kg(output_paths['distill_cpnet']['csv'], output_paths['distill_cpnet']['vocab'])
    
    routines = {
        'common': [
            {'func': construct_graph, 'args': (output_paths['distill_cpnet']['csv'], output_paths['distill_cpnet']['vocab'],
                                               output_paths['distill_cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['distill_cpnet']['csv'], output_paths['distill_cpnet']['vocab'],
                                               output_paths['distill_cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['distill_cpnet']['vocab'], output_paths['distill_cpnet']['patterns'])},
        ],
        'distill_csqa_eval': [
            {'func': convert_to_entailment, 'args': (input_paths['distill_csqa_eval']['distill'], output_paths['distill_csqa_eval']['statement']['distill'])},
            {'func': ground, 'args': (output_paths['distill_csqa_eval']['statement']['distill'], output_paths['distill_cpnet']['vocab'],
                                      output_paths['distill_cpnet']['patterns'], output_paths['distill_csqa_eval']['grounded']['distill'], nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['distill_csqa_eval']['grounded']['distill'], output_paths['distill_cpnet']['pruned-graph'], output_paths['distill_cpnet']['vocab'], output_paths['distill_csqa_eval']['graph']['adj-distill'], nprocs)},
        ],'distill_obqa_eval': [
            {'func': convert_to_entailment, 'args': (input_paths['distill_obqa_eval']['distill'], output_paths['distill_obqa_eval']['statement']['distill'])},
            {'func': ground, 'args': (output_paths['distill_obqa_eval']['statement']['distill'], output_paths['distill_cpnet']['vocab'],
                                      output_paths['distill_cpnet']['patterns'], output_paths['distill_obqa_eval']['grounded']['distill'], nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['distill_obqa_eval']['grounded']['distill'], output_paths['distill_cpnet']['pruned-graph'], output_paths['distill_cpnet']['vocab'], output_paths['distill_obqa_eval']['graph']['adj-distill'], nprocs)},
        ]
        }
        
    for rt in ['common', 'distill_obqa_eval']:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run.')
    
def preprocess_func():
    nprocs = 1

    if(os.path.exists("/home/lpz/MHL/KG_Distillation/GreaseLM/data/distill_obqa/statement/distill.statement.jsonl-sl100.loaded_cache")):
        os.remove("/home/lpz/MHL/KG_Distillation/GreaseLM/data/distill_obqa/statement/distill.statement.jsonl-sl100.loaded_cache")
    if(os.path.exists("/home/lpz/MHL/KG_Distillation/GreaseLM/data/distill_obqa/graph/distill.graph.adj.pk-nodenum200.loaded_cache")):
        os.remove("/home/lpz/MHL/KG_Distillation/GreaseLM/data/distill_obqa/graph/distill.graph.adj.pk-nodenum200.loaded_cache")

    routines = {
        'distill_obqa': [
            {'func': convert_to_entailment, 'args': (input_paths['distill_obqa']['distill'], output_paths['distill_obqa']['statement']['distill'])},
            {'func': ground, 'args': (output_paths['distill_obqa']['statement']['distill'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['distill_obqa']['grounded']['distill'], nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['distill_obqa']['grounded']['distill'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['distill_obqa']['graph']['adj-distill'], nprocs)},
        ]}

    for rt_dic in routines['distill_obqa']:
        rt_dic['func'](*rt_dic['args'])

    print('Successfully run.')
'''
if __name__ == '__main__':
    main()
    # pass
'''