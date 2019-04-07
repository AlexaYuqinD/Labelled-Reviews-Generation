import dataset as mydataset
from typing import List, Tuple
from model import CVAE
from train import train
import argparse
import torch as T
import tqdm
import utils as ut
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--label_cats', type=int, default=5)
parser.add_argument('--max_words', type=int, default=20000)
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=512)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--save_iter', type=int, default=10000)
parser.add_argument('--log_iter', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--embedding', type=str, default='glove.6B.300d')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--dataset', type=str, default='amazon')
parser.add_argument('--eval_model', type=str, default='CVAE_amazon_20181123-202309')
parser.add_argument('--eval_model_iter', type=int, default=200000)
parser.add_argument('--mask_prob', type=float, default=0.05)
parser.add_argument('--latent_max', type=int, default=1000, help="Only used in latent plotting")

opt = parser.parse_args()
opt.feature_dim = opt.emb_size + opt.label_cats + 1
cls_dict = {i : str(i) for i in range(5)}

raw_data : List[Tuple[List[str], str, float]] = mydataset.tokenize_data(opt.dataset)
raw_data = list(filter(lambda x : len(x[0]) <= opt.max_len, raw_data))
print("Load %d datapoints." % len(raw_data))

train_data, val_data, vocab = mydataset.make_dataset(raw_data, opt)
opt.max_words = min(opt.max_words, len(vocab) - 4)
train_iter, val_iter = mydataset.make_iterator((train_data, val_data), opt)
device = T.device(opt.device)

start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

layout = [
    ('model={:s}',  'cvae'),
    ('z={:02d}',  opt.z_dim),
    ('time={:s}', start_time),
    ('data={:s}', opt.dataset)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
writer = ut.prepare_writer(model_name, overwrite_existing=False)

model = CVAE(opt).to(device)
model.embedding.weight.data.copy_(vocab.vectors).to(device)

train_gen = mydataset.make_loader(train_iter, opt)
val_gen = mydataset.make_loader(val_iter, opt)
if opt.mode == 'train':
    train(model=model,
          train_loader = train_gen,
          val_loader = val_gen,
          tqdm=tqdm.tqdm,
          device=device,
          writer=writer,
          start_time=start_time,
          dataset_name=opt.dataset,
          iter_max=opt.max_iter,
          iter_log=opt.log_iter,
          iter_save=opt.save_iter,
          mask_prob=opt.mask_prob)
elif opt.mode == 'test':
    ut.load_model_by_name(model, opt.eval_model, global_step=opt.eval_model_iter)
    model.eval()
    os.makedirs(os.path.join("result", opt.dataset), exist_ok=True)
    with open(os.path.join("result", opt.dataset, "result_1.txt"), "w") as handle:
        for i in tqdm.tqdm(range(100)):
            generated_list = model.generate_samples(opt, vocab, samples=10)
            for ind, sentence in enumerate(generated_list):
                handle.write("%s, %d\n" % (sentence, ind % 5))
                # handle.write("%s\n" % sentence)
                handle.flush()
    # generated_list = model.generate_samples(opt, vocab)
    # for ind, sentence in enumerate(generated_list):
    #     print(sentence)
elif opt.mode == 'latent':
    def init_figure():
        plt.figure()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect('equal', adjustable='box')
    ut.load_model_by_name(model, opt.eval_model, global_step=opt.eval_model_iter)
    model.eval()
    os.makedirs(os.path.join("plots", opt.dataset), exist_ok=True)
    results = defaultdict(list)
    is_full = set()
    i = 0
    for x, y in train_gen:
        if len(is_full) > 1 and len(is_full) == len(results):
            break
        i += 1
        z = model.encode(x.to(device), y.to(device)).cpu().data.numpy()
        x_np = x.cpu().data.numpy()
        y_np = y.cpu().data.numpy()
        for cls, latent in zip(y_np, z):
            cls = cls[0]
            if len(results[cls]) < opt.latent_max:
                results[cls].append(latent)
            elif len(results[cls]) == opt.latent_max:
                is_full.add(cls)
    latent_list = []
    pca = PCA(n_components=2)
    init_figure()
    for cls, latent in results.items():
        latent = np.stack(latent, axis=0)
        latent_list.append(latent)
        latent_r = pca.fit_transform(latent)
        for x, y in latent_r:
            plt.scatter(x, y, color='navy', s=8)
        plt.title("Encoded latent variables of class %s" % cls_dict[cls])
        plt.savefig(os.path.join("plots", opt.dataset, str(cls) + ".png"))
        init_figure()
    latent = pca.fit_transform(np.concatenate(latent_list, axis=0))
    for x, y in latent:
        plt.scatter(x, y, color='navy', s=8)
    plt.title("Encoded latent variables of all class")
    plt.savefig(os.path.join("plots", opt.dataset, "all.png"))



