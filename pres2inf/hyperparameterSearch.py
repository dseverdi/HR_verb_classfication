from CNNTextClassifier import *
from utils import *
from trainModel import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

def cross_validate(model, train_set, output_path, batch_size, lr, nbOfFolds=10):
    #nbOfFolds = 10
    train_loss_ukupno = 0
    val_loss_ukupno = 0
    val_acc_ukupno = 0
    val_rmse_ukupno = 0
    brojac = 1
    #print(len(list(kf.split(train_set))))
    kfold = StratifiedKFold(n_splits=nbOfFolds, shuffle=True, random_state=1)
    X = [x[0] for x in train_set]
    y = [y[1] for y in train_set]
    for train, test in kfold.split(X,y):
        #output_dir = "results/{:%Y%m%d_%H%M%S}/fold{}/".format(datetime.now(),brojac)
        #output_path = output_dir + "model.weights"
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        print('fold:',brojac)
        train_loss, val_loss, val_acc, val_rmse = train_model(model, train_set[train[0]:train[-1]+1], train_set[test[0]:test[-1]+1], output_path, batch_size = batch_size, epochs=500, lr=lr)
        train_loss_ukupno += train_loss
        val_loss_ukupno += val_loss
        val_acc_ukupno += val_acc
        val_rmse_ukupno += val_rmse
        brojac += 1
        #print(len(test))
        #print(len(train_set[test[0]:test[-1]+1]))
        #print("%s %s" % (train, train_set[test[0]:test[-1]+1]))
    print("nakon svih foldova: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (train_loss_ukupno/nbOfFolds, val_loss_ukupno/nbOfFolds, val_acc_ukupno/nbOfFolds, val_rmse_ukupno/nbOfFolds))
    return train_loss_ukupno/nbOfFolds, val_loss_ukupno/nbOfFolds, val_acc_ukupno/nbOfFolds, val_rmse_ukupno/nbOfFolds

# u rjecniku hiperparametri su hiperparametri poslagani: batch_size, filter_sizes, num_filters, dropout_rate, learning_rate
hiperparametri = {'hp1':[25, [1, 2, 3, 5], 36, 0.1, 0.005], 'hp2':[50, [1, 2, 3, 5], 36, 0.1, 0.005], 'hp3':[25, [1, 2, 3, 5], 7, 0.1, 0.005], 'hp4':[50, [1, 2, 3, 5], 7, 0.1, 0.005],\
                 'hp5':[25, [1, 2, 3], 7, 0.1, 0.005], 'hp6':[50, [1, 2, 3], 7, 0.1, 0.005], 'hp7':[25, [1, 2, 3, 5], 15, 0.1, 0.005], 'hp8':[50, [1, 2, 3, 5], 15, 0.1, 0.005], \
                 'hp9':[25, [1, 2, 3], 15, 0.1, 0.005],'hp10':[50, [1, 2, 3], 15, 0.1, 0.005]}

best = 1e10
najbolji_path = ''
train_set, val_set, test_set, all_categories = loadData('../')
out_size=len(all_categories)

ft = loadFastText('../../')
weights_matrix = getWeightsMatrix(char_list,ft)

train_set_cv = train_set
train_set_cv.extend(val_set)

for hp in hiperparametri:
    file = open('outputi.txt','a')
    print('batch_size: %d, filter_sizes: %s, num_filters: %d, dropout_rate: %f, learning_rate: %f'%(hiperparametri[hp][0], str(hiperparametri[hp][1]), hiperparametri[hp][2],hiperparametri[hp][3],hiperparametri[hp][4]))
    file.write('batch_size: %d, filter_sizes: %s, num_filters: %d, dropout_rate: %f, learning_rate: %f \n'%(hiperparametri[hp][0], str(hiperparametri[hp][1]), hiperparametri[hp][2], hiperparametri[hp][3], hiperparametri[hp][4]))
    model = CNN_Text(out_size, vocab_len, 300, weights_matrix, hiperparametri[hp][1], hiperparametri[hp][2], hiperparametri[hp][3])
    model = model.to("cuda")
    output_dir = "results/hyperparameters/{}/".format(hp)
    print(output_dir)
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tl, vl, vacc, vrmse = cross_validate(model, train_set_cv, output_path, batch_size = hiperparametri[hp][0], lr=hiperparametri[hp][4])
    file.write("nakon svih foldova: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f \n \n" % (tl, vl, vacc, vrmse))
    file.close()
    if vl < best:
        best = vl
        najbolji_path = output_path
#file_out = open('najbolji_path.txt','w')        
#print(najbolji_path)
#file_out.write(najbolji_path)
#file_out.close()