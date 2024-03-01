# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: francisco camargo
"""

import pandas as pd
import sklearn.model_selection as ms
import sklearn.preprocessing as pp


def make_features(project_alias, preprocess_dict):

    filename        = preprocess_dict['filename']
    raw_dir         = preprocess_dict['raw_dir']
    interim_dir     = preprocess_dict['interim_dir']
    target_column   = preprocess_dict['target_column']
    names           = preprocess_dict['headers']

    test_size       = preprocess_dict['test_size']
    random_state    = preprocess_dict['random_state']
    shuffle         = preprocess_dict['shuffle']
    stratify        = preprocess_dict['stratify']

    save_pkl        = preprocess_dict['save_pkl']

    # Import raw data
    try:
        df = pd.read_csv(raw_dir+filename, names=names)
    except:
        df = pd.read_csv('../../'+raw_dir+filename, names=names)

    # Rename target column to 'y' (if necessary)
    if target_column: df.rename(columns={target_column: 'y'}, inplace=True)

    # Replace '?' with 'q': for mushroom data
    # df.replace('?', 'q', inplace=True)

    # Data summary stats
    print()
    print('Pre-Processing:', project_alias)
    print('\tSummary stats:')
    ljust = 19
    print('\t\tNum features'.ljust(ljust)+':', str(df.shape[1]-1).rjust(4))
    print('\t\tNum samples'.ljust(ljust)+':', df.shape[0])
    print('\t\tNum target labels'.ljust(ljust)+':', str(len(set(df['y']))).rjust(4))

    # Drop features with only a single value; provides no information
    try:
        drop_mask = df.describe().loc['unique'] == 1
        df = df.loc[:, ~drop_mask] # keep columns not in drop_mask
        dropped_columns = list(drop_mask[drop_mask].index)
        print('\tColumns dropped for having only one unique value:')
        print('\t\t',dropped_columns)
    except:
        pass

    # Distribution of target class labels
    class_dist = df['y'].value_counts()/df['y'].value_counts().sum()
    print('\tTarget label distribution:')
    for idx, val in class_dist.items():
        print('\t\t', str(idx).ljust(6)+':', str(round(val*100,1)).rjust(4))

    # Encoding of data
    if project_alias == 'mushroom': # have both categorical and ordinal data
        # Binary features/target
        list_binary     = ['y', 'bruises', 'gill-size', 'stalk-shape']
        binary_encoder  = pp.OrdinalEncoder(dtype='int')
        binary_array    = binary_encoder.fit_transform(df[list_binary])
        binary_array    = -1*(2*binary_array - 1)
            # set target convention as: poisonous to -1 and edible to 1
            # don't care about the sign of the binary features
        df_binary       = pd.DataFrame(binary_array, columns=list_binary)
        df.drop(columns=list_binary, inplace=True) # Drop binary columns

        # Ordinal features
        dict_ordinal = {
            'gill-spacing': ['c', 'w', 'd'],
            'ring-number' : ['n', 'o', 't'],
            'population'  : ['y', 'v', 's', 'n', 'c', 'a']
            }
        list_ordinal    = [my_list for my_list in dict_ordinal.values()]
        ordinal_encoder = pp.OrdinalEncoder(categories=list_ordinal, dtype='int')
        ordinal_array   = ordinal_encoder.fit_transform(df[dict_ordinal])
        df_ordinal      = pd.DataFrame(ordinal_array, columns=dict_ordinal)
        df.drop(columns=dict_ordinal, inplace=True) # Drop ordinal columns

        # Categorical features
        encoder     = pp.OneHotEncoder(sparse=False, dtype='int')
        cat_array   = encoder.fit_transform(df)
        df_cat      = pd.DataFrame(cat_array)

        # Concatenate all data
        df = pd.concat([df_binary, df_ordinal, df_cat], axis=1)

    elif project_alias == 'car': # All data is ordinal
        buying_list     = ['low', 'med', 'high', 'vhigh']
        maint_list      = ['low', 'med', 'high', 'vhigh']
        doors_list      = ['2', '3', '4', '5more']
        persons_list    = ['2', '4', 'more']
        lug_boot_list   = ['small', 'med', 'big']
        safety_list     = ['low', 'med', 'high']
        y_list          = ['unacc', 'acc', 'good', 'vgood']
        list_of_lists   = [buying_list, maint_list, doors_list, persons_list, lug_boot_list, safety_list, y_list]
        encoder         = pp.OrdinalEncoder(categories=list_of_lists, dtype='int')
        df              = pd.DataFrame(encoder.fit_transform(df), columns=df.columns)

    elif project_alias == 'cancer':
        df.drop(columns=['id', 'Unnamed: 32'], inplace = True)
        df['y'].replace({'B': -1, 'M': 1}, inplace=True)

    # Train-Test Splitting
    stratify = df['y'] if stratify else None
    df_train, df_test = ms.train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify)

    # Export to .pkl files
    if save_pkl:
        my_string = interim_dir+project_alias+'_interim'
        try:
            df.to_pickle(my_string+'.pkl')
            df_train.to_pickle(my_string+'_train.pkl')
            df_test.to_pickle( my_string+'_test.pkl')
        except:
            df.to_pickle('../../'+my_string+'.pkl')
            df_train.to_pickle('../../'+my_string+'_train.pkl')
            df_test.to_pickle( '../../'+my_string+'_test.pkl')

    return df_train, df_test


def main():
    config_list = ['../../configs/config_mushroom.yaml', '../../configs/config_car.yaml']
    config_list = ['../../configs/config_mushroom.yaml']
    # config_list = ['../../configs/config_car.yaml']
    # config_list = ['../../configs/config_monk.yaml']
    # config_list = ['../../configs/config_cancer.yaml']
    for config_file in config_list:
        config_dict = ry.read_yaml(config_file)
        df_train, df_test = make_features(config_dict['project_alias'], config_dict['preprocess'])


if __name__ == "__main__":

    import sys
    if '..' not in sys.path:
        sys.path.append('..')
    import read_yaml as ry

    main()
