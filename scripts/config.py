feature_templates = {
    't0': ['chr_position', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
           'plus1', 'plus2', 'plus3', 'plus4', 'next2letters',
           'prev2letters', 'prev1_word_suffix', 'following1_word_prefix',
           'focus_word_prefix', 'focus_word_suffix'],

    't1': ['chr_position', 'minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
                 'plus1', 'plus2', 'plus3', 'plus4', 'plus5', 'next2letters', 
                 'prev2letters', 'prev1_word_suffix', 'following1_word_prefix',
                 'focus_word_prefix', 'focus_word_suffix'],

    't2': ['minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus', 'plus1', 'plus2', 'plus3',
           'plus4', 'plus5', 'next2letters', 'prev2letters', 'focus_word_prefix', 'focus_word_suffix',
           'prev1_word_prefix', 'prev2_word_prefix', 'prev3_word_prefix',
           'prev1_word_suffix', 'prev2_word_suffix', 'prev3_word_suffix',
           'following1_word_prefix', 'following2_word_prefix', 'following3_word_prefix',
           'following1_word_suffix', 'following2_word_suffix', 'following3_word_suffix'],

    't3': ['chr_position', 'minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus', 'plus1', 'plus2', 'plus3', 'plus4', 'plus5',
           'next2letters', 'prev2letters', 'next3letters', 'prev3letters', 'next4letters', 'prev4letters', 'next5letters', 'prev5letters',
           'focus_word_prefix', 'focus_word_suffix',
           'prev1_word_prefix', 'prev2_word_prefix', 'prev3_word_prefix',
           'prev1_word_suffix', 'prev2_word_suffix', 'prev3_word_suffix',
           'following1_word_prefix', 'following2_word_prefix', 'following3_word_prefix',
           'following1_word_suffix', 'following2_word_suffix', 'following3_word_suffix'],

    't4': ['chr_position', 'minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus', 'plus1', 'plus2', 'plus3', 'plus4', 'plus5',
           'next2letters', 'prev2letters', 'next3letters', 'prev3letters', 'next4letters', 'prev4letters',
           'next5letters', 'prev5letters',
           'focus_word_prefix', 'focus_word_suffix',
           'prev1_word_prefix', 'prev2_word_prefix', 'prev3_word_prefix',
           'prev1_word_suffix', 'prev2_word_suffix', 'prev3_word_suffix',
           'following1_word_prefix', 'following2_word_prefix', 'following3_word_prefix',
           'following1_word_suffix', 'following2_word_suffix', 'following3_word_suffix']
}

categorical_indices = {
    't1': 'all'
}

catboost_config = {
    'default': {
        'iterations': 300,
        'thread_count': 10,
        'early_stopping_rounds': 20,
        'logging_level': 'Verbose'
    }
}

lightgbm_config = {
    'default': {
        'num_iterations': 400,
        'num_threads': 10,
        'early_stopping_rounds': 10,
        'train_metric': True
    }
}