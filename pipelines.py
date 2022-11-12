from pipeline import Pipeline
import data_utils as du
import linreg_utils as lu

# Работа с данными --------------------------------------------------------------------------------------------------

def transform_data(values=['X', 'Y']):
    if(type(values) != list):
        values = [values]
    pipe = Pipeline('transform_data')
    for V in values:
        pipe.transform(V, [V, V+'_min', V+'_max'], du.minmax_features, p_enable='_minmax')
        pipe.transform(V, [V, V+'_mean', V+'_std'], du.std_scale_features, p_enable='_std')
    return pipe


def revert_transform(values):
    if(type(values) != list):
        values = [values]
    pipe = Pipeline('revert_transform')
    for (V, V_old) in values:
        pipe.transform([V, V_old+'_min', V_old+'_max'], V, du.revert_minmax, p_enable='_minmax')
        pipe.transform([V, V_old+'_mean', V_old+'_std'], V, du.revert_std_scale, p_enable='_std')
    return pipe


def prepare_data(X='X', Y='Y'):
    pipe = Pipeline('prepare_data')
    pipe.transform(X, X, du.to_mat)
    pipe.transform(X, X, du.prepare_features, poly="_poly")
    if(len(Y)):
        pipe.transform(Y, Y, du.to_mat)
    return pipe


def split_test_train(X='X', Y='Y'):
    pipe = Pipeline('split_test_train')
    pipe.transform([X, Y], [X+'_train', X+'_test', Y+'_train', Y+'_test'],
        du.split_test_train, train_percent="_train_percent")
    return pipe

def load_train(X='X', Y='Y'):
    return Pipeline('train').transform([X+'_train', Y+'_train'], [X, Y], (lambda X, Y: (X, Y)))

def load_test(X='X', Y='Y'):
    return Pipeline('test').transform([X+'_test', Y+'_test'], [X, Y], (lambda X, Y: (X, Y)))

# Модель --------------------------------------------------------------------------------------------------

def predict(X='X', Y='Y'):
    pipe = Pipeline('activation')
    pipe.transform([X, 'W'], Y+'_pred', lu.predict)
    pipe.transform(Y+'_pred', Y+'_pred', lu.sigmoid, p_enable="_act_sigmoid")
    return pipe

def metrics(Y='Y'):
    pipe = Pipeline('metrics')
    pipe.transform([Y+'_pred', Y], 'MSE', lu.MSE, store_history=True, p_enable="_MSE")
    pipe.transform([Y+'_pred', Y], 'RMSE', lu.RMSE, store_history=True, p_enable="_RMSE")
    pipe.transform([Y+'_pred', Y], 'MAPE', lu.MAPE, store_history=True, p_enable="_MAPE")

    pipe.transform([Y+'_class_pred', Y], ['tp', 'tn', 'fn', 'fp'], lu.confusion_matrix, p_enable="_conf_mat")
    pipe.transform(['tp', 'tn', 'fn', 'fp'], 'accuracy', lu.accuracy, p_enable="_conf_mat")
    pipe.transform(['tp', 'fn'], 'recall', lu.recall, p_enable="_conf_mat")
    pipe.transform(['tp', 'fp'], 'precision', lu.precision, p_enable="_conf_mat")
    pipe.transform(['tp', 'fp', 'fn'], 'f1_measure', lu.f1_measure, p_enable="_conf_mat")
    return pipe

def loss(Y='Y'):
    pipe = Pipeline('loss')
    pipe.transform([Y+'_pred', Y, 'loss'], 'loss',
        lambda Y_p, Y_t, l: l + lu.MSE(Y_p, Y_t), p_enable="_loss_MSE")
    pipe.transform([Y+'_pred', Y, 'loss'], 'loss',
        lambda Y_p, Y_t, l: l + lu.BCE(Y_p, Y_t), p_enable="_loss_BCE")
    return pipe


def train_model(learn, X='X', Y='Y', name="", pred=predict(), **args):
    if type(learn) == list:
        learn = Pipeline.compose(learn, name)
    if type(predict) == list:
        learn = Pipeline.compose(predict, name)

    begin_iteration = Pipeline('begin_iteration')
    begin_iteration.subpipeline(predict('X_batch', 'Y_batch'))
    begin_iteration.subpipeline(loss('Y_batch'))
    begin_iteration.generate('der_loss', lambda: 0)

    end_iteration = Pipeline('end_iteration')
    end_iteration.transform(['W', 'der_loss'], 'W', lu.update_weights, lr="_lr")

    epoch = Pipeline('epoch')
    epoch.transform([X, Y], ['X_batches', 'Y_batches'], du.take_batches, batch_size="_batch")
    epoch.generate('loss', lambda: 0)

    epoch.subpipeline(Pipeline.compose([begin_iteration, learn, end_iteration]),
        p_repeat=[('X_batches', 'X_batch'), ('Y_batches', 'Y_batch')])

    epoch.transform(['loss', 'X_batches'], 'loss', lambda l, X_b: l/len(X_b), store_history=True)
    epoch.transform([X, 'W'], Y+'_pred', lu.predict)
    epoch.subpipeline(pred)
    epoch.subpipeline(metrics(Y))

    pipe = Pipeline('model')
    pipe.transform(X, 'W', lu.generate_weights)
    pipe.subpipeline(epoch, p_repeat='_epochs', **args)
    return pipe


def regularization():
    pipe = Pipeline('regularization')
    pipe.transform(['W', 'der_loss'], 'der_loss',
        lambda W, dl, l: dl + lu.der_ridge_reg_loss(W, l), p_enable='_ridge_l', l='_ridge_l')
    pipe.transform(['W', 'loss'], 'loss',
        lambda W, ls, l: ls + lu.ridge_reg_loss(W, l), p_enable='_ridge_l', l='_ridge_l')
    pipe.transform(['W', 'der_loss'], 'der_loss',
        lambda W, dl, l: dl + lu.der_lasso_reg_loss(W, l), p_enable='_lasso_l', l='_lasso_l')
    pipe.transform(['W', 'loss'], 'loss',
        lambda W, ls, l: ls + lu.lasso_reg_loss(W, l), p_enable='_lasso_l', l='_lasso_l')
    return pipe


def gradient_descent():
    pipe = Pipeline('gradient_descent')
    pipe.transform(['Y_batch_pred', 'X_batch', 'Y_batch', 'der_loss'], 'der_loss',
        lambda Y_p, X, Y, dl: dl + lu.grad_loss(Y_p, X, Y))
    return pipe

def classify_binary(X='X', Y='Y'):
    pipe = Pipeline('classification')
    pipe.subpipeline(predict(X, Y))
    pipe.transform([Y+'_pred'], Y+"_class_pred", lu.to_class, threshold="_threshold")
    return pipe