import torch
import numpy as np
from torch.nn.functional import softmax
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score


def eval_model(
    train, 
    test, 
    model, 
    target='reg',
    model_class='reduced-rank', 
    training_type='single-sess', 
    session_idx=None, 
):
    
    train_x, train_y = [], []
    for (x, y) in train:
        train_x.append(x.cpu())
        train_y.append(y.cpu())
        
    if training_type == 'multi-sess':
        train_x = torch.vstack(train_x)
        train_y = torch.vstack(train_y)
    else:
        train_x = torch.stack(train_x)
        train_y = torch.stack(train_y)

    test_x, test_y = [], []
    for (x, y) in test:
        test_x.append(x.cpu())
        test_y.append(y.cpu())
        
    if training_type == 'multi-sess':
        test_x = torch.vstack(test_x)
        test_y = np.vstack(test_y)
    else:
        test_x = torch.stack(test_x)
        test_y = np.stack(test_y)

    if model_class == 'reduced-rank':
        if training_type == 'multi-sess':
            assert session_idx is not None
            test_pred = model(test_x, session_idx)
        else:
            test_pred = model(test_x)
        if target == 'clf':
            test_pred = softmax(test_pred, dim=1).detach().numpy().argmax(1)
        else:
            test_pred = test_pred.detach().numpy()

    elif model_class == 'linear':
        train_x, test_x = train_x.numpy(), test_x.numpy()
        model.fit(train_x.reshape((train_x.shape[0], -1)), train_y.argmax(1))
        test_pred = model.predict(test_x.reshape((test_x.shape[0], -1)))
        
    elif model_class in ['mlp', 'lstm']:
        test_pred = model(test_x)
        if target == 'clf':
            test_pred = softmax(test_pred, dim=1).detach().numpy().argmax(1)
        else:
            test_pred = test_pred.detach().numpy()
        
    else:
        raise NotImplementedError

    if target == 'reg':
        metric = r2_score(test_y.flatten(), test_pred.flatten())
    elif target == 'clf':
        metric = accuracy_score(test_y.argmax(1), test_pred)
        
    return metric, test_pred, test_y


def eval_multi_session_model(
    train_lst, 
    test_lst, 
    model, 
    target='reg',
    model_class='reduced-rank', 
):
    metric_lst, test_pred_lst, test_y_lst = [], [], []
    for idx, (train, test) in enumerate(zip(train_lst, test_lst)):
        metric, test_pred, test_y = eval_model(
            train, test, model, target='reg', 
            model_class=model_class, training_type='multi-sess',
            session_idx=idx
        )
        metric_lst.append(metric)
        test_pred_lst.append(test_pred)
        test_y_lst.append(test_y)
    return metric_lst, test_pred_lst, test_y_lst
    
