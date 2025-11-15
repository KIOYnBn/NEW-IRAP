# import packages
import os
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
from sklearn import svm
from joblib import dump, load
import Load as iload
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV


# train
def svm_train(file, c_number, gamma, out):
    train_data, train_label = iload.load_svmfile(file)
    my_svm = svm.SVC(kernel='rbf', C=c_number, gamma=gamma)
    model = my_svm.fit(train_data, train_label)
    dump(model, out)


# predict
def svm_predict(file, model, out=False):
    test_data, test_label = iload.load_svmfile(file)
    model = load(model)
    predict_label = model.predict(test_data)
    if out != False:
        result = 'True,Predict\n'
        for i in range(len(test_label)):
            result += str(test_label[i]) + ',' + str(predict_label[i]) + '\n'
        svm_acc(test_label, predict_label)
        with open(out, 'w', encoding="utf-8") as f:
            f.write(result)
    else:
        return test_label, predict_label


# evaluate
def svm_evaluate(file, c_number, gamma, cv):
    test_data, test_label = iload.load_svmfile(file)
    my_svm = svm.SVC(kernel='rbf', C=c_number, gamma=gamma)
    predict_label = cross_val_predict(my_svm, test_data, test_label, cv=cv)
    return test_label, predict_label


# grid search
def svm_grid(file):
    my_svm = svm.SVC(decision_function_shape="ovo", random_state=0)
    c_number = []
    for i in range(-5, 15 + 1, 2):
        c_number.append(2 ** i)
    gamma = []
    for i in range(-15, 3 + 1, 2):
        gamma.append(2 ** i)
    parameters = {'C': c_number, 'gamma': gamma}
    new_svm = GridSearchCV(my_svm, parameters, cv=5, scoring="accuracy", return_train_score=False, n_jobs=1)
    train_data, train_label = iload.load_svmfile(file)
    model = new_svm.fit(train_data, train_label)
    best_c = model.best_params_['C']
    best_g = model.best_params_['gamma']
    return best_c, best_g


# grid search for folder
def svm_grid_folder(path, out=False):
    out_box = {}
    for i in path:
        out_box[i] = svm_grid(i)
        if out != False:
            file = i + '\tC_numbr: ' + str(round(out_box[i][0], 4)) + '\tGamma: ' + str(round(out_box[i][1], 4)) + '\n'
            with open(out, 'a') as f:
                f.write(file)
    if out == False:
        return out_box


# create grid file for folder
def svm_set_hys(path, c=8, g=0.125, out=False):
    out_box = {}
    for i in os.listdir(path):
        out_box[i] = (c, g)
        if out != False:
            file = os.path.join(path, i) + '\tC_numbr: ' + str(c) + '\tGamma: ' + str(g) + '\n'
            with open(out, 'a') as f:
                f.write(file)
    if out == False:
        return out_box


# acc
def svm_acc(y_t, y_p):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    y_t = list(y_t)
    y_p = list(y_p)
    for i in range(len(y_t)):
        t_label = y_t[i]
        p_label = y_p[i]
        if t_label == p_label == 0:
            tp += 1
        if t_label == p_label == 1:
            tn += 1
        if t_label == 0 and p_label == 1:
            fn += 1
        if t_label == 1 and p_label == 0:
            fp += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    print('\nPredict result: ' + str(round(acc, 3)*100) + '%')