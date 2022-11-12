import numpy as np

def predict(X, W):
    return X * W.T

def least_squares(X, Y):
    return (np.linalg.inv((X.T * X)) * X.T * Y).T

def MSE(Y, Y_true):
    return np.mean(np.power(Y - Y_true, 2))

def RMSE(Y, Y_true):
    return np.sqrt(MSE(Y, Y_true))

def MAPE(Y, Y_true):
    return np.mean(np.fabs((Y_true - Y) / Y_true)) * 100

def BCE(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def generate_weights(X):
    return np.matrix(np.random.randn(X.shape[1])/np.sqrt(np.sum(X.shape[1])))

def grad_loss(Y, X_t, Y_t):
    return ((Y - Y_t).T * X_t) / X_t.shape[0]

def update_weights(B, loss, lr):
    return B - lr * loss

def ridge_reg_loss(B, l):
    return (l/2)*np.sum(B[1:])

def der_ridge_reg_loss(B, l):
    term = 2 * B[:, 1:] * l
    return np.concatenate([np.zeros((1, 1)), term], axis=1)

def lasso_reg_loss(B, l):
    return l*np.sum(np.abs(B[1:]))

def der_lasso_reg_loss(B, l):
    term = np.sign(B[:, 1:]) * l
    return np.concatenate([np.zeros((1, 1)), term], axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))     

def to_class(logit, threshold):
    return (logit>=threshold)*1

def confusion_matrix(Y, Y_true):
    tp = 0 # true_positives
    tn = 0 # true_negatives
    fp = 0 # false_positives
    fn = 0 # false_negatives

    for yhati,yi in zip(Y, Y_true):
        if yi == 1 and yhati == 1:
            tp += 1
        elif yi == 0 and yhati == 0:
            tn += 1
        elif yi == 1 and yhati == 0:
            fn += 1
        elif yi == 0 and yhati == 1:
            fp += 1

    return tp, tn, fn, fp

def accuracy(tp, tn, fn, fp):
    if tp + tn + fn + fp:
        return (tp + tn) / (tp + tn + fn + fp)
    else:
        return 0

def recall(tp, fn):
    if tp+fn:
        return tp / (tp + fn)
    else:
        return 0

def precision(tp, fp):
    if tp + fp:
        return tp / (tp + fp)
    else:
        return 0

def f1_measure(tp, fp, fn):
    if tp + fp + fn:
        return tp / (tp + 0.5*(fp+fn))
    else:
        return 0
