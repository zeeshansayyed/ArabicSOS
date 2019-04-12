feature_templates = {
    't1': ['chr_position', 'minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
                 'plus1', 'plus2', 'plus3', 'plus4', 'plus5', 'next2letters', 
                 'prev2letters', 'prev_word_suffix', 'following_word_prefix',
                 'focus_word_prefix', 'focus_word_suffix'],
    't2': ['minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
                 'plus1', 'plus2', 'plus3', 'plus4', 'plus5',
                   'prev_word_minus1', 'prev_word_minus2', 'prev_word_minus3',
                  'following_word_plus0', 'following_word_plus1', 'following_word_plus2']
}

categorical_indices = {
    't1': 'all'
}

catboost_config = {
    'default': {
        'iterations': 200,
        'thread_count': 4,
        'early_stopping_rounds': 10,
        'logging_level': 'Verbose'
    }
}

lightgbm_config = {
    'default': {
        'num_iterations': 50,
        'num_threads': 4,
        'early_stopping_rounds': 10,
        'train_metric': True
    }
}