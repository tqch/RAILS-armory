from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from armory import paths
import os, re


class ASKLoss(nn.Module):
    """
    Adversarial Soft K-nearest neighbor loss
    """

    def __init__(
            self,
            reduction="mean",
            temperature=1,
            metric="l2",
            type="instance-wise"
    ):
        super(ASKLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.metric = metric
        self.type = type

    @staticmethod
    def pairwise_l2_distance(x, y, x_other=None):
        """
        x_i,y_j in R^D
        for i=1..M,j=1..N, calculate ||x_i-y_j||^2 => O(3*M*N*D)
        is equivalent to
          for i=1..M ||x_i||^2
        + for i=1..M,j=1..N <x_i,y_j>
        + for j=1..N ||y_j||^2
        _______________________
        O(2*(M+M*N+N)*D+2*M*N)
        """
        dist_matrix = (x.pow(2).sum(dim=1)[:, None] - 2 * x @ y.T + y.pow(2).sum(dim=1)[None, :]).sqrt()
        dist_orig = None
        if x_other is not None:
            dist_orig = (x - x_other).pow(2).sum(dim=1).sqrt()
        return dist_matrix, dist_orig

    @staticmethod
    def pairwise_cosine_similarity(x, y, x_other=None):
        similarity_matrix = x @ y.T / y.pow(2).sum(dim=1)[None, :].sqrt()
        similarity_matrix = 1 / x.pow(2).sum(dim=1)[:, None].sqrt() * similarity_matrix
        similarity_orig = None
        if x_other is not None:
            similarity_orig = (x * x_other).sum(dim=1) / \
                              x.pow(2).sum(dim=1).sqrt() / x_other.pow(2).sum(dim=1).sqrt()
        return similarity_matrix, similarity_orig

    def forward(self, x, y, x_ref, y_ref, x_other=None):
        if x_ref.ndim == 2:
            if self.metric == "l2":
                score_matrix, score_orig = -self.pairwise_l2_distance(x, x_ref, x_other)
            if self.metric == "cosine":
                score_matrix, score_orig = self.pairwise_cosine_similarity(x, x_ref, x_other)
        elif x_ref.ndim == 3:
            score_orig = None
            if self.metric == "l2":
                score_matrix = -(x.unsqueeze(1) - x_ref).pow(2).sum(dim=-1).sqrt()
                if x_other is not None:
                    score_orig = -(x - x_other).pow(2).sum(dim=-1).sqrt()
            if self.metric == "cosine":
                score_matrix = torch.zeros(x.size(0), x_ref.size(1)).to(x)
                for i in range(x.size(0)):
                    score_matrix[i, :] += x[i] @ x_ref[i].T / x[i].pow(2).sum().sqrt()\
                                          / x_ref[i].pow(2).sum(dim=-1).sqrt()
                if x_other is not None:
                    score_orig = (x * x_other).sum(dim=-1) / x.pow(2).sum(dim=-1).sqrt() / x_other.pow(2).sum(dim=-1).sqrt()
        soft_nns = torch.zeros(x.size(0), 10).to(x)
        if self.type == "instance-wise":
            if score_orig is not None:
                score_matrix = F.softmax(torch.cat([
                    score_matrix, score_orig[:, None]
                ], dim=1) / self.temperature, dim=1)
            else:
                score_matrix = F.softmax(score_matrix / self.temperature, dim=1)
            for i in range(10):
                if (y_ref == i).sum().item() == 0:
                    soft_nns[:, i] += 1e-6
                else:
                    if score_orig is not None:
                        soft_nns[:, i] += score_matrix[:, :-1][:, y_ref == i].sum(dim=1) + 1e-6
                    else:
                        soft_nns[:, i] += score_matrix[:, y_ref == i].sum(dim=1) + 1e-6
            if score_orig is not None:
                soft_nns[range(x.size(0)), y] += score_matrix[:, -1]
            log_soft_nns = torch.log(soft_nns)
            return F.nll_loss(log_soft_nns, y, reduction=self.reduction)
        elif self.type == "class-wise":
            if score_orig is not None:
                score_matrix = torch.cat([
                    score_matrix, score_orig[:, None]
                ], dim=1) / self.temperature
            else:
                score_matrix = score_matrix / self.temperature
            class_size = []
            for i in range(10):
                class_size.append((y_ref == i).sum())
                if score_orig is not None:
                    soft_nns[:, i] += score_matrix[:, :-1][:, y_ref == i].sum(dim=1)
                else:
                    soft_nns[:, i] += score_matrix[:, y_ref == i].sum(dim=1)
            if score_orig is not None:
                soft_nns[range(x.size(0)), y] += score_matrix[:, -1]
            soft_nns = soft_nns / torch.tensor(class_size)[None, :].to(x)
            if score_orig is not None:
                factor = torch.tensor(class_size).to(x)[y_ref]
                soft_nns[range(x.size(0)), y] *= factor / (factor + 1)
            return F.cross_entropy(soft_nns, y, reduction=self.reduction)


class ASKAttack:
    def __init__(
            self,
            model,
            n_class=10,
            n_neighbors=5,
            class_samp_size=None,
            eps=8 / 255,
            step_size=2 / 255,
            max_iter=10,
            random_init=True,
            metric="l2",
            batch_size=20,
            hidden_layers=-1,
            kappa=1,
            temperature=0.1,
            random_seed=1234
    ):
        self.class_samp_size = class_samp_size
        self.n_class = n_class
        self.metric = metric
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        
        train_data, train_targets = model.train_data, model.train_targets
        query_objects = model.query_objects
        self._model = self._wrap_model(model if callable(model) else model._model._model._model)
        
        self.device = next(iter(self._model.parameters())).device
        self._model.eval()

        self.hidden_layers = self._model.hidden_layers
        self.random_seed = random_seed
        
        self.train_data = self._samp_data(train_data, train_targets)
        self.n_neighbors = n_neighbors
        if query_objects is None:
            self._nns = self._build_nns()
        else:
            self._nns = [[] for _ in range(len(self.hidden_layers))]
            for i, layer in enumerate(self.hidden_layers):
                qob = query_objects[str(layer)]
                self._nns[i] = [
                    AnnoyIndex(f=qob["f"], metric=qob["metric"])
                    for _ in range(self.n_class)
                ]
                for q,fp in zip(self._nns[i], qob["paths"]):
                    q.load(os.path.join(qob["saved_object_dir"], fp))

        self.temperature = [temperature for _ in range(len(self.hidden_layers))] \
            if isinstance(temperature, (int, float)) else temperature
        self.kappa = [kappa for _ in range(len(self.hidden_layers))] \
            if isinstance(kappa, (int, float)) else kappa

        self.ask_loss = [ASKLoss(temperature=t, metric=metric, type="class-wise") for t in self.temperature]

        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_init = random_init

    def _samp_data(
            self,
            train_data,
            train_targets,
    ):
        if self.class_samp_size is None:
            return [train_data[train_targets == i] for i in range(self.n_class)]
        else:
            np.random.seed(self.random_seed)
            class_indices = []
            for i in range(self.n_class):
                inds = np.where(train_targets == i)[0]
                subset = np.random.choice(inds, size=self.class_samp_size, replace=False)
                class_indices.append(subset)
            return [train_data[subset] for subset in class_indices]

    def _get_hidden_repr(self, x, return_targets=False):
        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i:i + self.batch_size]
            with torch.no_grad():
                if return_targets:
                    hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
                else:
                    hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            if self.metric == "cosine":
                hidden_reprs_batch = [
                    hidden_repr_batch/hidden_repr_batch.pow(2).sum(dim=-1,keepdim=True).sqrt()
                    for hidden_repr_batch in hidden_reprs_batch
                ]
            hidden_reprs_batch = [hidden_repr_batch.cpu() for hidden_repr_batch in hidden_reprs_batch]
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            torch.cat([hidden_batch[i] for hidden_batch in hidden_reprs], dim=0)
            for i in range(len(self.hidden_layers))
        ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.hidden_mappings = []
                start_layer = 0
                if hasattr(model, "feature"):
                    start_layer = 1
                    self.hidden_mappings.append(model.feature)
                self.hidden_mappings.extend([
                    m[1] for m in model.named_children()
                    if isinstance(m[1], nn.Sequential) and re.match("^f\d$", m[0]) is not None
                ])
                if hidden_layers == -1:
                    self.hidden_layers = list(range(len(self.hidden_mappings)))
                else:
                    self.hidden_layers = hidden_layers
                self.hidden_layers = [hl + start_layer for hl in hidden_layers]
                self.classifier = self._model.classifier

            def forward(self, x):
                if x.size(1) != 3:
                    x = x.permute(0,3,1,2)
                hidden_reprs = []
                for mp in self.hidden_mappings:
                    x = mp(x)
                    hidden_reprs.append(x)
                out = self.classifier(x if x.ndim == 3 else x.flatten(start_dim=1))
                return [hidden_reprs[i].flatten(start_dim=1) for i in self.hidden_layers], out

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        nns = [[] for _ in range(len(self.hidden_layers))]
        for class_data in self.train_data:
            hidden_reprs, _ = self._get_hidden_repr(class_data)
            for i, hidden_repr in enumerate(hidden_reprs):
                f = len(hidden_repr[0])
                
                if self.metric == "cosine":
                    query_object = AnnoyIndex(f, 'angular')
                else:
                    query_object = AnnoyIndex(f, 'euclidean')
                for j in range(len(class_data)):
                    query_object.add_item(j, hidden_repr[j])
                query_object.build(n_trees=20, n_jobs=-1)
                nns[i].append(query_object)
        return nns

    def attack(self, x, y, x_refs, x_adv=None):
        if x_adv is None:
            if self.random_init:
                x_adv = 2 * self.eps * (torch.rand_like(x) - 0.5) + x
                x_adv = x_adv.clamp(0.0, 1.0)
            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        hidden_repr_adv, _ = self._model(x_adv)
        loss = 0
        for ask_loss, hidden_repr, x_ref, kappa in zip(self.ask_loss, hidden_repr_adv, x_refs, self.kappa):
            if self.metric == "cosine":
                hidden_repr = hidden_repr / hidden_repr.pow(2).sum(dim=1, keepdim=True).sqrt()
            loss += kappa * ask_loss(
                hidden_repr,
                y,
                x_ref.to(x),
                torch.arange(self.n_class).repeat_interleave(self.n_neighbors).to(x)
            )
        grad = torch.autograd.grad(loss, x_adv)[0]
        pert = self.step_size * grad.sign()
        x_adv = (x_adv + pert).clamp(0.0, 1.0).detach()
        pert = (x_adv - x).clamp(-self.eps, self.eps)
        return x + pert

    def _get_nns(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        nn_reprs = []
        for i, hidden_repr, nns in zip(range(len(self.hidden_layers)), hidden_reprs, self._nns):
            #nn_inds = [torch.LongTensor(nn.kneighbors(hidden_repr, return_distance=False)) for nn in nns]
            nn_inds = [torch.LongTensor([nn.get_nns_by_vector(hidden_repr[j], self.n_neighbors, search_k=-1,\
                                                include_distances=False) for j in range(len(x))]) for nn in nns]
            nn_repr = [class_data[nn_ind] for class_data, nn_ind in zip(self.train_data, nn_inds)]
            nn_reprs.append(self._get_hidden_repr(torch.cat(nn_repr, dim=1).reshape(-1, *x.shape[1:]))[0][i])
        return [nn_repr.reshape(x.size(0), self.n_neighbors*self.n_class, -1) for nn_repr in nn_reprs]

    def generate(self, x, y=None):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            nn_reprs_batch = self._get_nns(x_batch)
            if y is None:
                y_batch = self._model(x_batch)
                if isinstance(y_batch, tuple):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(self.device)
            else:
                y_batch = y[i: i + self.batch_size].to(self.device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(x_batch, y_batch, nn_reprs_batch)
                else:
                    x_adv_batch = self.attack(x_batch, y_batch, nn_reprs_batch, x_adv_batch)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu().numpy()
