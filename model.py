import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

class Highway(nn.Module):
    def __init__(self, opt):
        super(Highway, self).__init__()
        self.n_layers = opt.n_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(opt.emb_size, opt.emb_size) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(opt.emb_size, opt.emb_size) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(opt.emb_size, opt.emb_size) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = T.sigmoid(self.gate[layer](x))	        #Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))	#Compute non linear information
            linear = self.linear[layer](x)	                #Compute linear information
            x = gate*non_linear + (1-gate)*linear           #Combine non linear and linear information according to gate

        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.highway = Highway(opt)
        self.fc = nn.Linear(opt.feature_dim, opt.emb_size)
        self.n_hidden_E = opt.z_dim
        self.n_layers_E = 1
        self.lstm = nn.LSTM(input_size=opt.emb_size, hidden_size=self.n_hidden_E, num_layers=self.n_layers_E, batch_first=True, bidirectional=True)
        self.device = T.device(opt.device)

    def init_hidden(self, batch_size):
        h_0 = T.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E).to(self.device)
        c_0 = T.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E).to(self.device)
        self.hidden = (h_0, c_0)

    def forward(self, x, y):
        batch_size, n_seq, n_embed = x.size()
        x = self.highway(self.fc(T.cat([x, y], dim=-1)))
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x, self.hidden)	             #Exclude c_T and extract only h_T
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        self.hidden = self.hidden[-1]	                             #Select only the final layer of h_T
        e_hidden = T.cat(list(self.hidden), dim=1)	                 #merge hidden states of both directions; check size
        return e_hidden

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.n_hidden_G = opt.z_dim
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        self.lstm = nn.LSTM(input_size=opt.feature_dim + self.n_z,
                            hidden_size=self.n_hidden_G, num_layers=self.n_layers_G, batch_first=True)
        self.fc = nn.Linear(self.n_hidden_G, opt.max_words + 4)
        self.device = T.device(opt.device)

    def init_hidden(self, batch_size):
        h_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G).to(self.device)
        c_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G).to(self.device)
        self.hidden = (h_0, c_0)

    def forward(self, x, y, z, g_hidden = None):
        batch_size, n_seq, n_embed = x.size()
        z = T.cat([z]*n_seq, 1).view(batch_size, n_seq, self.n_z)	#Replicate z inorder to append same z at each time step
        x = T.cat([x, y, z], dim=2)

        if g_hidden is None:	                                    #if we are validating
            self.init_hidden(batch_size)
        else:					                                    #if we are training
            self.hidden = g_hidden

        #Get top layer of h_T at each time step and produce logit vector of vocabulary words
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)

        return output, self.hidden	                                #Also return complete (h_T, c_T) incase if we are testing


class CVAE(nn.Module):
    def __init__(self, opt):
        super(CVAE, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.max_words + 4, opt.emb_size)
        self.encoder = Encoder(opt)
        self.hidden_to_mu = nn.Linear(2*opt.z_dim, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2*opt.z_dim, opt.n_z)
        self.generator = Generator(opt)
        self.n_z = opt.n_z
        self.device = T.device(opt.device)

    def forward(self, x, y, z = None, G_hidden = None):
        if z is None:	                                                #If we are testing with z sampled from random noise
            batch_size, n_seq = x.size()
            x = self.embedding(x)	                                    #Produce embeddings from encoder input
            y = y.unsqueeze(1).expand(-1, x.shape[1], y.shape[1])
            E_hidden = self.encoder(x, y)	                            #Get h_T of Encoder
            mu = self.hidden_to_mu(E_hidden)	                        #Get mean of lantent z
            logvar = self.hidden_to_logvar(E_hidden)	                #Get log variance of latent z
            z = T.randn([batch_size, self.n_z]).to(self.device)         #Noise sampled from ε ~ Normal(0,1)
            z = mu + z*T.exp(0.5*logvar)	                            #Reparameterization trick: Sample z = μ + ε*σ for backpropogation
            kld = -0.5*T.sum(logvar-mu.pow(2)-logvar.exp()+1, 1).mean()	#Compute KL divergence loss
        else:
            kld = None                                                  #If we are training with given text

        logit, G_hidden = self.generator(x, y, z, G_hidden)
        return logit, G_hidden, kld

    def loss(self, x, y, G_inp, cur_step = 0, is_train=True):
        logit, _, kld = self.forward(x, y, G_inp, None, None)
        logit = logit.view(-1, self.opt.max_words + 4)  # converting into shape (batch_size*(n_seq-1), n_vocab) to facilitate performing F.cross_entropy()
        # x = x[:, 1:x.size(1)]  # target for generator should exclude first word of sequence
        x = x.contiguous().view(-1)  # converting into shape (batch_size*(n_seq-1),1) to facilitate performing F.cross_entropy()
        rec_loss = F.cross_entropy(logit, x)
        kld_coef = (math.tanh((cur_step - 15000) / 1000) + 1) / 2
        loss = self.opt.rec_coef * rec_loss + kld_coef * kld

        prefix = "train" if is_train else "test"
        summaries = dict((
            ('%s/loss' % prefix, loss),
            ('%s/elbo' % prefix, -(rec_loss + kld)),
            ('%s/kl_z' % prefix, kld),
            ('%s/rec' % prefix, rec_loss),
        ))

        return loss, summaries

    def generate_samples(self, opt, vocab, samples = 10, max_len = 100):
        device = T.device(opt.device)
        str_list = []
        for i in range(samples):
            z = T.randn([1, opt.n_z]).to(device)
            h_0 = T.zeros(opt.n_layers_G, 1, opt.z_dim).to(device)
            c_0 = T.zeros(opt.n_layers_G, 1, opt.z_dim).to(device)
            G_hidden = (h_0, c_0)
            G_inp = T.LongTensor(1, 1).fill_(vocab.stoi['<sos>']).to(device)
            str = "<sos> "
            if opt.dataset == 'pitchfork':
                features = np.concatenate((np.eye(10), np.repeat([5], 10).reshape((-1, 1))), axis=1)[i % 10].reshape((1, 1, -1))
            elif opt.dataset == 'sentiment':
                features = np.array(5).reshape((1, 1, 1))
            elif opt.dataset == "amazon":
                features = np.concatenate((np.eye(5), np.repeat([1], 5).reshape((-1, 1))), axis=1)[i % 5].reshape((1, 1, -1))
            else:
                raise AttributeError
            features = T.from_numpy(features.astype(np.float32)).to(device)
            word_count = 0
            while G_inp[0][0].item() != vocab.stoi['<eos>']:
                if word_count > max_len:
                    break
                word_count += 1
                with T.autograd.no_grad():
                    logit, G_hidden, _ = self.forward(None, features, z, G_hidden)
                probs = F.softmax(logit[0], dim=1)
                G_inp = T.multinomial(probs, 1)
                if vocab.itos[G_inp[0][0].item()] == "<pad>":
                    continue
                str += (vocab.itos[G_inp[0][0].item()] + " ")
            str_list.append(str.encode('utf-8'))
        return str_list

    def encode(self, x, y):
        batch_size, n_seq = x.size()
        x = self.embedding(x)	                                    #Produce embeddings from encoder input
        y = y.unsqueeze(1).expand(-1, x.shape[1], y.shape[1])
        E_hidden = self.encoder(x, y)	                            #Get h_T of Encoder
        mu = self.hidden_to_mu(E_hidden)	                        #Get mean of lantent z
        logvar = self.hidden_to_logvar(E_hidden)	                #Get log variance of latent z
        z = T.randn([batch_size, self.n_z]).to(self.device)         #Noise sampled from ε ~ Normal(0,1)
        z = mu + z*T.exp(0.5*logvar)	                            #Reparameterization trick: Sample z = μ + ε*σ for backpropogation
        return z

