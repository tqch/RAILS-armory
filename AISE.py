import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
import math,time
from collections import Counter
import gc

class GenAdapt:
    '''
    core component of AISE B-cell generation
    '''

    def __init__(self, mut_range, mut_prob, mode='random'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.mode = mode

    def crossover(self, base1, base2, select_prob):
        assert base1.ndim == 2 and base2.ndim == 2, "Number of dimensions should be 2"
        crossover_mask = torch.rand_like(base1) < select_prob[:,None]
        return torch.where(crossover_mask, base1, base2)

    def mutate_random(self, base):
        mut = 2 * torch.rand_like(base) - 1  # uniform (-1,1)
        mut = self.mut_range * mut
        mut_mask = torch.rand_like(base) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, 1)

    def crossover_complete(self, parents, select_prob):
        parent1,parent2 = parents
        child = self.crossover(parent1,parent2,select_prob)
        child = self.mutate_random(child)
        return child

    def __call__(self, *args):
        if self.mode == "random":
            base, *_ = args
            return self.mutate_random(base)
        elif self.mode == "crossover":
            assert len(args) == 2
            parents, select_prob = args
            return self.crossover_complete(parents,select_prob)
        else:
            raise ValueError("Unsupported mutation type!")

class MyTimer:
    def __init__(self,title="default title"):
        self.title = title
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    def __exit__(self,*exc_args):
        self.end_time = time.perf_counter()
        print("{} ellapse time: {}".format(self.title,self.end_time-self.start_time))
        
            
class L2NearestNeighbors(NearestNeighbors):
    '''
    compatible query object class for euclidean distance
    '''

    def __call__(self, X):
        return self.kneighbors(X, return_distance=False)

    
def neg_l2_dist(x,y):
    '''
    x: (1,n_feature)
    y: (N,n_feature)
    '''
    return -(x-y).pow(2).sum(dim=1).sqrt()

def inner_product(X, Y):
    return (X@Y.T)[0]

class AISE:
    '''
    implement the Adaptive Immune System Emulation
    '''

    def __init__(self, x_orig, y_orig, hidden_layer=None, model=None, input_shape=None, device=torch.device("cuda"),
                 n_class=10, n_neighbors=10, query_class="l2", norm_order=2, normalize=False,
                 avg_channel=False, fitness_function="negative l2", sampling_temperature=.3, adaptive_temp=False,
                 max_generation=50, requires_init=True, apply_bound="none", c=1.0,
                 mut_range=(.05, .15), mut_prob=(.05, .15), mut_mode="crossover",
                 decay=(.9, .9), n_population=1000, memory_threshold=.25, plasma_threshold=.05, 
                 keep_memory=False, return_log=True):

        self.model = model
        self.device = device

        self.x_orig = x_orig
        self.y_orig = y_orig
        
        if input_shape is None:
            try:
                self.input_shape = tuple(self.x_orig.shape[1:])  # mnist: (1,28,28)
            except:
                print("Invalid data type for x_orig!")
        else:
            self.input_shape = input_shape
        
        self.hidden_layer = hidden_layer

        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self.query_class = query_class
        self.norm_order = norm_order
        self.normalize = normalize
        self.avg_channel = avg_channel
        self.fitness_func = self._get_fitness_func(fitness_function)
        self.sampl_temp = sampling_temperature
        self.adaptive_temp = adaptive_temp
        
        self.max_generation = max_generation
        self.n_population = self.n_class * self.n_neighbors
        self.requires_init = requires_init
        self.apply_bound = apply_bound
        self.c = c
        
        self.mut_range = mut_range
        self.mut_prob = mut_prob

        if isinstance(mut_range, float):
            self.mut_range = (mut_range, mut_range)
        if isinstance(mut_prob, float):
            self.mut_prob = (mut_prob, mut_prob)

        self.mut_mode = mut_mode
        self.decay = decay
        self.n_population = n_population
        self.n_plasma = int(plasma_threshold*self.n_population)
        self.n_memory = int(memory_threshold*self.n_population)-self.n_plasma
        
        self.keep_memory = keep_memory
        self.return_log = return_log

        try:
            self.model.to(self.device)
            self.model.eval()
        except:
            print("Invalid model!")
        
        try:
            self._query_objects = self._build_all_query_objects()
        except:
            print("Cannot build query objects!")
        
    def _get_fitness_func(self,func_str):
        if func_str == "negative l2":
            return neg_l2_dist
        elif func_str == "inner product":
            return inner_product

    def _build_class_query_object(self, xh_orig, class_label=-1):
        if class_label + 1:
            x_class = xh_orig[self.y_orig == class_label]
        else:
            x_class = xh_orig
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors,n_jobs=-1).fit(x_class)
        return query_object

    def _build_all_query_objects(self):
        xh_orig = self._hidden_repr_mapping(self.x_orig,query=True).detach().cpu().numpy()
        if self.adaptive_temp:
            self.sampl_temp *= np.sqrt(xh_orig.shape[1]/np.prod(self.input_shape)).item()  # heuristic sampling temperature: proportion to the square root of feature space dimension
        if self.n_class:
            print("Building query objects for {} classes {} samples...".format(self.n_class, self.x_orig.size(0)),
                  end="")
            query_objects = [self._build_class_query_object(xh_orig,class_label=i) for i in range(self.n_class)]
            print("done!")
        else:
            print("Building one single query object {} samples...".format(self.x_orig.size(0)), end="")
            query_objects = [self._build_class_query_object(xh_orig)]
            print("done!")
        return query_objects
        
    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            print("Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, len(Q)),
                  end="")
            rel_ind = [query_obj(Q) for query_obj in self._query_objects]
            abs_ind = []
            for c in range(self.n_class):
                class_ind = np.where(self.y_orig.numpy() == c)[0]
                abs_ind.append(class_ind[rel_ind[c]])
            print("done!")
        else:
            print("Searching {} naive B cells for each of {} antigens...".format(self.n_neighbors, Q.size(0)),
                  end="")
            abs_ind = [query_obj(Q) for query_obj in self._query_objects]
            print('done!')
        return abs_ind

    def _hidden_repr_mapping(self, x, batch_size=2048, query=False):
        '''
        transform b cells and antigens into inner representations of AISE
        '''
        if self.hidden_layer is not None:
            xhs = []
            for i in range(0,x.size(0),batch_size):
                xx = x[i:i+batch_size]
                with torch.no_grad():
                    if query:
                        xh = self.model.truncated_forward(self.hidden_layer)(xx.to(self.device)).detach().cpu()
                    else:
                        xh = self.model.truncated_forward(self.hidden_layer)(xx.to(self.device))
                    if self.avg_channel:
                        xh = xh.sum(dim=1)
                    xh = xh.flatten(start_dim=1)
                    if self.normalize:
                        xh = xh/xh.pow(2).sum(dim=1,keepdim=True).sqrt()
                    xhs.append(xh.detach())
            return torch.cat(xhs)
        else:
            xh = x.flatten(start_dim=1)
            if self.normalize:
                xh = xh/xh.pow(2).sum(dim=1,keepdim=True).sqrt()
            return xh.detach()

    def clip_class_bound(self,x,y,class_center,class_bound):
        return torch.min(torch.max(x,(class_center-class_bound)[y]),(class_center+class_bound)[y])
        
    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        pla_bcs = []
        pla_labs = []
        if self.keep_memory:
            mem_bcs = []
            mem_labs = []
        else:
            mem_bcs = None
            mem_labs = None
        print("Affinity maturation process starts with population of {}...".format(self.n_population))
        ant_logs = []  # store the history dict in terms of metrics for antigens
        for n in range(ant.size(0)):
            # print(torch.cuda.memory_summary())
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], mode=self.mut_mode)
            curr_gen = torch.cat([self.x_orig[ind[n]].flatten(start_dim=1) for ind in nbc_ind]).to(self.device)  # naive b cells
            labels = np.concatenate([self.y_orig[ind[n]] for ind in nbc_ind])
            if self.apply_bound != "none":
                class_center = []
                if self.apply_bound == "hard":
                    class_bound = []
                for i in range(0,len(curr_gen),self.n_neighbors):
                    class_center.append(torch.mean(curr_gen[i:i+self.n_neighbors],dim=0))
                    if self.apply_bound == "hard":
                        class_bound.append((curr_gen[i:i+self.n_neighbors]-class_center[-1]).abs().max(dim=0)[0])
                class_center = torch.stack(class_center)
                if self.apply_bound == "hard":
                    class_bound = torch.stack(class_bound)            
            if self.requires_init:
                assert self.n_population % (
                        self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = np.tile(labels, self.n_population // (self.n_class * self.n_neighbors))
                if self.apply_bound == "hard":
                    curr_gen = self.clip_class_bound(curr_gen,labels,class_center,class_bound)
            curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
            fitness_score = self.fitness_func(ant_tran[n].unsqueeze(0).to(self.device), curr_repr.to(self.device))
            if self.apply_bound == "soft":
                fitness_score = fitness_score + self.c*F.pairwise_distance(curr_gen,class_center[labels])
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            ant_log = dict()  # history log for each antigen
            # zeroth generation logging
            fitness_pop_hist = []
            pop_fitness = fitness_score.sum().item()
            fitness_pop_hist.append(pop_fitness)
            if y_ant is not None:
                fitness_true_class_hist = []
                pct_true_class_hist = []
                true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                fitness_true_class_hist.append(true_class_fitness)
                true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                pct_true_class_hist.append(true_class_pct)
            
            static_index = torch.LongTensor(torch.arange(len(labels))).to(self.device)
            for i in range(self.max_generation):
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                parents_ind1 = Categorical(probs=survival_prob).sample((self.n_population,))    
                if self.mut_mode == "crossover":
                    parents_ind2 = torch.zeros_like(parents_ind1)
                    for c in range(self.n_class):
                        pos = static_index[labels[parents_ind1.cpu()]==c]
                        if len(pos):
                            parents_ind2_class = Categorical(probs=F.softmax(fitness_score[static_index[labels==c]] / self.sampl_temp,dim=-1)).sample((len(pos),))
                            parents_ind2[pos] = static_index[labels==c][parents_ind2_class.cpu()]
                    parent_pairs = [curr_gen[parents_ind1],curr_gen[parents_ind2]]
                    curr_gen = genadapt(parent_pairs, fitness_score[parents_ind1] /\
                                      (fitness_score[parents_ind1]+fitness_score[parents_ind2]))
                else:
                    parents = curr_gen[parents_ind1]
                    curr_gen = genadapt(parents)
                
                if self.apply_bound == "hard":
                    curr_gen = self.clip_class_bound(curr_gen,labels,class_center,class_bound) 
                
                curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
                labels = labels[parents_ind1.cpu()]

                fitness_score = self.fitness_func(ant_tran[n].unsqueeze(0).to(self.device), curr_repr.to(self.device))
                if self.apply_bound == "soft":
                    fitness_score = fitness_score + self.c*F.pairwise_distance(curr_gen,class_center[labels])
                pop_fitness = fitness_score.sum().item()

                # logging
                fitness_pop_hist.append(pop_fitness)
                if y_ant is not None:
                    true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                    fitness_true_class_hist.append(true_class_fitness)
                    true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                    pct_true_class_hist.append(true_class_pct)

                # check homogeneity
                if len(np.unique(labels)) == 1:
                    break # early stop

                # adaptive shrinkage of certain hyper-parameters
                if self.decay:
                    assert len(self.decay) == 2
                    if pop_fitness < best_pop_fitness:
                        if num_plateau >= max(math.log(self.mut_range[0] / self.mut_range[1], self.decay[0]),
                                              math.log(self.mut_prob[0] / self.mut_prob[1], self.decay[1])):
                            # early stop
                            break
                        decay_coef = tuple(decay_coef[i] * self.decay[i] for i in range(2))
                        num_plateau += 1
                        genadapt = GenAdapt(max(self.mut_range[0], self.mut_range[1] * decay_coef[0]),
                                            max(self.mut_prob[0], self.mut_prob[1] * decay_coef[1]),
                                            mode=self.mut_mode)
                    else:
                        best_pop_fitness = pop_fitness
            _, fitness_rank = torch.sort(fitness_score.cpu())
            ant_log["fitness_pop"] = fitness_pop_hist
            if y_ant is not None:
                ant_log["fitness_true_class"] = fitness_true_class_hist
                ant_log["pct_true_class"] = pct_true_class_hist
            pla_bcs.append(curr_gen[fitness_rank[-self.n_plasma:]].detach().cpu())
            pla_labs.append(labels[fitness_rank[-self.n_plasma:]])
            if self.keep_memory:
                mem_bcs.append(curr_gen[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]].detach().cpu())
                mem_labs.append(labels[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]])
            ant_logs.append(ant_log)        
        
        pla_bcs = torch.stack(pla_bcs).view((-1,self.n_plasma)+self.input_shape).numpy()
        pla_labs = np.stack(pla_labs).astype(np.int)
        if self.keep_memory:
            mem_bcs = torch.stack(mem_bcs).view((-1,self.n_mem)+self.input_shape).numpy()
            mem_labs = np.stack(mem_labs).astype(np.int)
        
        return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs

    def clonal_expansion(self, ant, y_ant=None):
        print("Clonal expansion starts...")
        ant_tran = self._hidden_repr_mapping(ant.detach())
        try:
            nbc_ind = self._query_nns_ind(ant_tran.detach().cpu().numpy())
        except:
            print("The object needs to be re-instaniated after one call!")
        mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = self.generate_b_cells(ant.flatten(start_dim=1), ant_tran,
                                                                               nbc_ind)
        if self.keep_memory:
            print("{} plasma B cells and {} memory generated!".format(pla_bcs.shape[0]*self.n_plasma, mem_bcs.shape[0]*self.n_memory))
        else:
            print("{} plasma B cells generated!".format(pla_bcs.shape[0]*self.n_plasma))
        if self.return_log:
            return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs
        else:
            return mem_bcs, mem_labs, pla_bcs, pla_labs
        
    def __call__(self, ant):
        _,_,_,pla_labs,*_ = self.clonal_expansion(ant)
        # delete all the data reference in the object after clonal expansion
        del self.x_orig
        del self.y_orig
        del self._query_objects
        gc.collect()
        # output the prediction of aise
        return AISE.predict_proba(pla_labs,self.n_class)
        
    @staticmethod
    def predict(labs,n_class):
        return AISE.predict_proba(labs,n_class).argmax(axis=1)

    @staticmethod
    def predict_proba(labs,n_class):
        return np.stack(list(map(lambda x: np.bincount(x,minlength=n_class)/x.shape[1], labs)))