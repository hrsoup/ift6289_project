import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

from train import evaluation_train
from test import evaluation_test
from sklearn.metrics import classification_report
from music_preprocess import X_music, Y_music, X_test, Y_test, X_valid, Y_valid, note2id
from utils import args

if args[1] == 'train':
    from language_en_plain_preprocess import X_language_pretrain, Y_language_pretrain, word2id
    language_type = args[2]

    X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language_pretrain, Y_language_pretrain, test_size=0.1, random_state=26)
    eval = evaluation_train(word2id, language_type)
    eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id)

elif args[1] == 'test':
    from language_en_plain_preprocess import X_language_original, Y_language_original, word2id
    X_language_en = deepcopy(X_language_original)
    Y_language_en = deepcopy(Y_language_original)
    word2id_en = word2id


    experiment_number = 1

    f1score_supervised_finetune_1 = []
    f1score_supervised_finetune_2 = []
    f1score_supervised_finetune_3 = []
    f1score_supervised_finetune_4 = []
    f1score_baseline = []    
    for i in range(experiment_number):


#--------------------------finetune_1--------------------------------------#

        X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language_en, Y_language_en, test_size=0.1)
        language_type = args[2]
        eval = evaluation_test(word2id_en, language_type)
        # eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id_en)
        y_pred, y_true = eval.test(X_test, Y_test, word2id_en) 

        # flatten list
        # print(y_pred)
        y_pred_list = [item for sublist in y_pred for item in sublist[1:]]
        y_true_list = [int(item) for sublist in y_true for item in sublist[1:]]

        # calculate f1 score 
        f1score_supervised_finetune_1.append(classification_report(y_true_list, y_pred_list, output_dict=True)['1']['f1-score'])

    print('jokes_music')
    for i in range(experiment_number+1):
        if i != experiment_number:
            print(format(f1score_supervised_finetune_1[i]))
        # elif i == experiment_number:
        #     print(format(np.mean(f1score_supervised_finetune_1), '.4f'))
