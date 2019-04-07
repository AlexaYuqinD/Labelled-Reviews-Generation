import numpy as np
import utils as ut
from torch import nn, optim
import torch as T

def train(model, train_loader, val_loader, tqdm, device, writer, start_time, dataset_name="pitchfork",
          iter_max=np.inf, iter_save=10000, iter_log=50, iter_val = 100, reinitialize=False,
          mask_prob = 0.0):
    # Optimization
    if reinitialize:
        try:
            model.reset_parameters()
        except AttributeError:
            pass
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            val_iter = iter(val_loader)
            for x, y in train_loader:
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                x = x.to(device).reshape(x.size(0), -1)
                y = y.to(device).reshape(x.size(0), -1)
                mask = (T.rand(x.shape).to(device) > mask_prob).long()
                x = x * mask
                G_inp = x[:, :-1]
                loss, summaries = model.loss(x, y, G_inp, i)

                loss.backward()
                optimizer.step()
                pbar.set_postfix(
                        loss='{:.3e}'.format(loss),
                        kl='{:.3e}'.format(summaries['train/kl_z']),
                        rec='{:.3e}'.format(summaries['train/rec']))
                pbar.update(1)

                if i % iter_val == 0:
                    x_val, y_val = next(val_iter)
                    x_val = x_val.to(device).reshape(x_val.size(0), -1)
                    y_val = y_val.to(device).reshape(y_val.size(0), -1)
                    G_inp = x_val[:, :-1]
                    loss, summaries = model.loss(x_val, y_val, G_inp, i, is_train=False)
                    ut.log_summaries(writer, summaries, i)

                # Log summaries
                if i % iter_log == 0:
                    ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, "_" + dataset_name + "_" + start_time, i)

                if i == iter_max:
                    return

