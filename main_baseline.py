import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

from full_music_evaluation import evaluation
from sklearn.metrics import classification_report
from music_preprocess import X_music, Y_music, X_test, Y_test, X_valid, Y_valid, note2id
from utils import args

if args[1] == 'train_test':

    experiment_number = 5

    f1score_supervised_finetune_1 = []
    f1score_supervised_finetune_2 = []
    f1score_supervised_finetune_3 = []
    f1score_supervised_finetune_4 = []
    f1score_baseline = []    
    for i in range(experiment_number):

        eval = evaluation(note2id)
        eval.train(X_music, Y_music, X_valid, Y_valid, note2id)
        y_pred, y_true = eval.test(X_test, Y_test, note2id) 

        # flatten list
        y_pred_list = [item for sublist in y_pred for item in sublist[1:]]
        y_true_list = [int(item) for sublist in y_true for item in sublist[1:]]

        # calculate f1 score 
        f1score_baseline.append(classification_report(y_true_list, y_pred_list, output_dict=True)['1']['f1-score'])
        print(classification_report(y_true_list, y_pred_list, output_dict=True)['1']['f1-score'])

    with open('output.txt', 'a') as file:

        print('full_music', file=file)
        for i in range(experiment_number+1):
            if i != experiment_number:
                print(format(f1score_baseline[i]), end=" & ", file=file)
            # elif i == experiment_number:
            #     print(format(np.mean(f1score_baseline)), file=file)
        
        print('\n', file=file)
