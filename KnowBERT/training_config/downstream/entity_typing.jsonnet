{
    "dataset_reader": {
        "type": "ultra_fine",
        "entity_masking": "entity_markers",
        "tokenizer_and_candidate_generator": {
                        "type": "bert_tokenizer_and_candidate_generator",
                        "entity_candidate_generators": {
                            "wordnet": {"type": "wordnet_mention_generator",
                                        "entity_file": "/home/lpz/mhl/KG_Distill/kb/kg_mine/entities.jsonl"},
                        },
                        "entity_indexers":  {
                            "wordnet": {
                                   "type": "characters_tokenizer",
                                   "tokenizer": {
                                       "type": "word",
                                       "word_splitter": {"type": "just_spaces"},
                                   },
                                   "namespace": "entity"
                                }
                        },
                        "bert_model_type": "bert-base-uncased",
                        "do_lower_case": true,
                    },
    },
    "iterator": {
        "iterator": {
            "type": "basic",
            "batch_size": 32
        },
        "type": "self_attn_bucket",
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "model": {
            "type": "from_archive",
            "archive_file": "/home/lpz/mhl/KG_Distill/kb/kb_mine/model.tar.gz",
        },
        "type": "simple-classifier",
        "bert_dim": 768,
        "concat_word_a": true,
        "include_cls": false,
        "metric_a": {
            "type": "f1_set"
        },
        "num_labels": 9,
        "task": "classification",
        "use_bce_loss": true
    },
    "train_data_path": "/home/lpz/mhl/KG_Distill/kb/datas/OpenEntity/train.json",
    "validation_data_path": "/home/lpz/mhl/KG_Distill/kb/datas/OpenEntity/dev.json",
    "trainer": {
        "cuda_device": 0,
        
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 10,
            "num_steps_per_epoch": 62.5
        },
        "num_epochs": 10,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "b2": 0.98,
            "lr": 3e-05,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+f1"
    },
    "vocabulary": {
        "directory_path": "/home/lpz/mhl/KG_Distill/kb/kg_mine/vocabulary_wordnet.tar.gz"
    }
}
