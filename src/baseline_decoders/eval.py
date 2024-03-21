import torch
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def eval_model(train, test, model, model_class='reduced_rank', plot=False):
    
    train_x, train_y = [], []
    for (x, y) in train:
        train_x.append(x.cpu())
        train_y.append(y.cpu())
    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    test_x, test_y = [], []
    for (x, y) in test:
        test_x.append(x.cpu())
        test_y.append(y.cpu())
    test_x = torch.stack(test_x)
    test_y = np.stack(test_y)

    if model_class == 'reduced_rank':
        test_pred = model(test_x).detach().numpy()

    elif model_class == 'reduced_rank_latents':
        U = model.U.cpu().detach().numpy()
        V = model.V.cpu().detach().numpy()

        train_proj_on_U = np.einsum('ktc,cr->ktr', train_x, U)
        test_proj_on_U = np.einsum('ktc,cr->ktr', test_x, U)
        weighted_train_proj = np.einsum('kdr,rdt->ktr', train_proj_on_U, V)
        weighted_test_proj = np.einsum('kdr,rdt->ktr', test_proj_on_U, V)

        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        regr = GridSearchCV(Ridge(), {'alpha': alphas})

        train_x, test_x = weighted_train_proj, weighted_test_proj
        regr.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        test_pred = regr.predict(test_x.reshape((test_x.shape[0], -1)))

    elif model_class == 'ridge':
        train_x, test_x = train_x.numpy(), test_x.numpy()
        model.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        test_pred = model.predict(test_x.reshape((test_x.shape[0], -1)))
        
    elif model_class in ['mlp', 'lstm']:
        test_pred = model(test_x).detach().numpy()

    else:
        raise NotImplementedError

    r2 = r2_score(test_y.flatten(), test_pred.flatten())

    if plot:
        plt.figure(figsize=(12, 2))
        plt.plot(test_y[:10].flatten(), c='k', linewidth=.5, label='target')
        plt.plot(test_pred[:10].flatten(), c='b', label='pred')
        plt.title(f"model: {model_class} R2: {r2: .3f}")
        plt.legend()
        plt.show()

    return r2, test_pred, test_y

