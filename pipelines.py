from pipeline import Pipeline
import data_utils as du
import linreg_utils as lu

def transform_data(values):
    if(type(values) != list):
        values = [values]
    pipe = Pipeline('transform_data')
    for V in values:
        pipe.transform(V, [V, V+'_min', V+'_max'], du.minmax_features, p_enable='_minmax')
        pipe.transform(V, [V, V+'_mean', V+'_std'], du.std_scale_features, p_enable='_std')
    return pipe

prepare_data = Pipeline('prepare_data')
prepare_data.transform('X', 'X', du.prepare_features, poly="_poly")
prepare_data.transform('Y', 'Y', du.prepare_target)

split_test_train = Pipeline('split_test_train')
split_test_train.transform(['X', 'Y'], ['X_train', 'X_test', 'Y_train', 'Y_test'], du.split_test_train, train_percent="_train_percent")
load_train = Pipeline('train').transform(['X_train', 'Y_train'], ['X', 'Y'], (lambda X, Y: (X, Y)))
load_test = Pipeline('test').transform(['X_test', 'Y_test'], ['X', 'Y'], (lambda X, Y: (X, Y)))

begin_iteration = Pipeline('begin_iteration')
begin_iteration.transform(['X_batch', 'W'], 'Y_batch_pred', lu.predict)
begin_iteration.generate('der_loss', lambda: 0)

gd_step = Pipeline('gd_step')
gd_step.transform(['Y_batch_pred', 'X_batch', 'Y_batch', 'der_loss'], 'der_loss', lambda Y_p, X, Y, dl: dl + lu.grad_loss(Y_p, X, Y))
gd_step.transform(['Y_batch_pred', 'Y_batch', 'loss'], 'loss', lambda Y_p, Y_t, l: l + lu.MSE(Y_p, Y_t))

regularization = Pipeline('regularization')
regularization.transform(['W', 'der_loss'], 'der_loss', lambda W, dl, l: dl + lu.der_ridge_reg_loss(W, l), p_enable='_ridge_l', l='_ridge_l')
regularization.transform(['W', 'loss'], 'loss', lambda W, ls, l: ls + lu.ridge_reg_loss(W, l), p_enable='_ridge_l', l='_ridge_l')
regularization.transform(['W', 'der_loss'], 'der_loss', lambda W, dl, l: dl + lu.der_lasso_reg_loss(W, l), p_enable='_lasso_l', l='_lasso_l')
regularization.transform(['W', 'loss'], 'loss', lambda W, ls, l: ls + lu.lasso_reg_loss(W, l), p_enable='_lasso_l', l='_lasso_l')

end_iteration = Pipeline('end_iteration')
end_iteration.transform(['W', 'der_loss'], 'W', lu.update_weights, lr="_lr")

metrics = Pipeline('metrics')
metrics.transform(['Y_pred', 'Y'], 'MSE', lu.MSE, store_history=True, p_enable="_MSE")
metrics.transform(['Y_pred', 'Y'], 'RMSE', lu.RMSE, store_history=True, p_enable="_RMSE")
metrics.transform(['Y_pred', 'Y'], 'MAPE', lu.MAPE, store_history=True, p_enable="_MAPE")

gd_epoch = Pipeline('gd_epoch')
gd_epoch.transform(['X', 'Y'], ['X_batches', 'Y_batches'], du.take_batches, batch_size="_batch")
gd_epoch.generate('loss', lambda: 0)
gd_epoch.subpipeline(Pipeline.compose([begin_iteration, gd_step, regularization, end_iteration]),
    p_repeat=[('X_batches', 'X_batch'), ('Y_batches', 'Y_batch')])
gd_epoch.transform(['loss', 'X_batches'], 'loss', lambda l, X_b: l/len(X_b), store_history=True)
gd_epoch.transform(['X', 'W'], 'Y_pred', lu.predict)
gd_epoch.subpipeline(metrics)



