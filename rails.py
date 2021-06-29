from __future__ import absolute_import, division, print_function, unicode_literals

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from aise import AISE
from dknn import DKNN
from models.vgg import VGG16
from art.classifiers import PyTorchClassifier
import logging
import numpy as np
import inspect
from armory import paths
from collections import deque

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAILSEvalWrapper(PyTorchClassifier):
    def __init__(self,**kwargs):
        super(RAILSEvalWrapper,self).__init__(**kwargs)
        self.train_data = None
        self.train_targets = None
        self.query_objects = None
        
    def _make_model_wrapper(self, model):
        return model

    def predict(self, x, batch_size=500, **kwargs):

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        
        mask_aise = False # whether to hide aise from attacks
        if self._model.attack_cnn:
            callers = [f.function for f in inspect.stack()]
            # check if the call is from attacker (for evaluation purpose only)
            if "generate" in callers:
                mask_aise = True
        if mask_aise:
            for m in range(num_batch):
                begin, end = (
                    m * batch_size,
                    min((m + 1) * batch_size, x_preprocessed.shape[0]),
                )
                with torch.no_grad():
                    output = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))[-1].cpu().numpy()
                results[begin:end] = output
            # Apply postprocessing
            predictions = self._apply_postprocessing(preds=results, fit=False)
            return predictions
        else:
            for m in range(num_batch):
                # Batch indexes
                begin, end = (
                    m * batch_size,
                    min((m + 1) * batch_size, x_preprocessed.shape[0]),
                )
                with torch.no_grad():
                    output = self._model.predict(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
                results[begin:end] = output
            # Apply postprocessing
            predictions = self._apply_postprocessing(preds=results, fit=False)
            return predictions

    
class RAILS(nn.Module):
    def __init__(
        self,
        model,
        train_data,
        train_targets,
        hidden_layers,
        query_objects,
        aise_params,
        batch_size=1024,
        n_class=None,
        start_layer=None,
        attack_cnn=False,
        use_dknn=False,
        state_dict=None
    ):    
        super(RAILS, self).__init__()
        if state_dict is not None:
            self.load_state_dict(state_dict)
        
        self.start_layer = start_layer or -1
        self.n_class = n_class or 10
        
        self._model = self.reconstruct_model(model, self.start_layer)
        
        self.batch_size = batch_size
        with torch.no_grad():
            self.train_data = torch.cat([
                self._model.to_start(train_data[i:i + self.batch_size].to(DEVICE)).cpu()
                for i in range(0, train_data.size(0), self.batch_size)
            ], dim=0).cpu()
        self.input_shape = self.train_data.shape[1:]
        self.train_targets = train_targets.cpu()
        
        self.query_objects = query_objects
        
        self.use_dknn = use_dknn
        self.attack_cnn = attack_cnn

        if not self.use_dknn:
            self.hidden_layers = hidden_layers
            self.aise_params = aise_params
            self.aise = []
            for i,layer in enumerate(self.hidden_layers):
                self.aise.append(AISE(
                    self.train_data,
                    self.train_targets,
                    query_objects=query_objects[str(layer)],
                    model=self._model, 
                    device=DEVICE, 
                    **self.aise_params[i]
                ))
        else:
            self.dknn = DKNN(
                model=self,
                train_data=train_data,
                y_train=train_targets,
                n_neighbors=5,
                device=DEVICE
            )
    
    def _make_layers(self, cfg, in_channels):
        layers = []
#         in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x        
        return nn.Sequential(*layers)

    def reconstruct_model(self, model, start_layer):

        class InternalModel(nn.Module):
            def __init__(self, model, start_layer=-1):
                super(InternalModel, self).__init__()
                self._model = model
                self.start_layer = start_layer
                self.feature_mappings = deque(
                    mod[1] for mod in self._model.named_children()
                    if not ("feature" in mod[0] or "classifier" in mod[0])
                )
                self.n_layers = len(self.feature_mappings)

                self.to_start = nn.Sequential()
                if hasattr(model, "feature"):
                    self.to_start.add_module(model.feature)
                for i in range(start_layer + 1):
                    self.to_start.add_module(
                        f"pre_start_layer{i}", self.feature_mappings.popleft()
                    )

                self.hidden_layers = range(self.n_layers-self.start_layer-1)

                self.truncated_forwards = [nn.Identity()]
                self.truncated_forwards.extend([
                    self._customize_mapping(hidden_layer)
                    for hidden_layer in self.hidden_layers
                ])

            def _customize_mapping(self, end_layer=None):
                feature_mappings = list(self.feature_mappings)[:end_layer + 1]

                def truncated_forward(x):
                    for map in feature_mappings:
                        x = map(x)
                    return x

                return truncated_forward

            def truncated_forward(self, hidden_layer):
                return self.truncated_forwards[hidden_layer - self.start_layer]

        return InternalModel(model, start_layer)
    
    def __call__(self,x):
        # check if channel first
        if x.size(1) != self.input_shape[0]:
            x = x.permute([0,3,1,2])
        return [self._model._model.forward(x),]
        
    def predict(self, x, batch_size=None):
        # check if channel first
        if x.size(1) != self.input_shape[0]:
            x = x.permute([0,3,1,2])
        if self.use_dknn:
            return self.dknn.predict(x)
        else:
            with torch.no_grad():
                x_start = torch.cat([
                    self._model.to_start(x[i:i + self.batch_size].to(DEVICE)).cpu()
                    for i in range(0, x.size(0), self.batch_size)
                ], dim=0)
            pred = np.zeros((x_start.size(0), self.n_class))
            for aise in self.aise:
                pred = pred + aise(x_start)
            return pred
        
    @property
    def get_layers(self):
        """
        Return the hidden layers in the model, if applicable.
        :return: The hidden layers in the model, input and output layers excluded.
        .. warning:: `get_layers` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this
                     is not guaranteed either. In addition, the function can only infer the internal
                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                     return the logit layer.
        """
        import torch.nn as nn

        result = []
        if isinstance(self, nn.Sequential):
            # pylint: disable=W0212
            # disable pylint because access to _modules required
            for name, module_ in self._modules.items():  # type: ignore
                result.append(name + "_" + str(module_))

        elif isinstance(self, nn.Module):
            result.append("final_layer")

        else:
            raise TypeError("The input model must inherit from `nn.Module`.")
        logger.info(
            "Inferred %i hidden layers on PyTorch classifier.", len(result),
        )

        return result


def make_rails_model(**kwargs):
    return RAILS(**kwargs)


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    
    assert weights_path is not None
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    cnn = VGG16()
    cnn.load_state_dict(checkpoint["state_dict"])
    cnn.to(DEVICE)
    cnn.eval()
    
    saved_object_dir = os.path.join(paths.runtime_paths().saved_model_dir,"query_objects")
    query_objects = checkpoint["query_objects"]
    for v in query_objects.values():
        v["saved_object_dir"] = saved_object_dir
    
    for params in cnn.parameters():
        params.requires_grad_(False)
    
    model = make_rails_model(
        model=cnn, 
        train_data=checkpoint["train_data"],
        train_targets=checkpoint["train_targets"],
        query_objects=query_objects,
        **model_kwargs
    )

    wrapped_model = RAILSEvalWrapper(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    
    wrapped_model.train_data = checkpoint["train_data"]
    wrapped_model.train_targets = checkpoint["train_targets"]
    wrapped_model.query_objects = query_objects
    
    return wrapped_model
