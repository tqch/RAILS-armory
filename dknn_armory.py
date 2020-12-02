from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepkNN import CKNN
from art.classifiers import PyTorchClassifier
import logging
import numpy as np
import inspect

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAILSEvalWrapper(PyTorchClassifier):
    def __init__(self,**kwargs):
        super(RAILSEvalWrapper,self).__init__(**kwargs)

    def _make_model_wrapper(self, model):
        return model

    def predict(self, x, batch_size=128, **kwargs):

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
                    output = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))[0].cpu().numpy()
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

      
        

    
class VGG(nn.Module):
    def __init__(self,train_data,train_targets,attack_cnn=False, state_dict=None):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
        cfg1 = [64, 64, 'M']
        cfg2 = [128, 128, 'M']
        cfg3 = [256, 256, 256, 'M']
        cfg4 = [512, 512, 512, 'M']
        cfg5 = [512, 512, 512, 'M']
        self.f1 = self._make_layers(cfg1, 3)
        self.f2 = self._make_layers(cfg2, 64)
        self.f3 = self._make_layers(cfg3, 128)
        self.f4 = self._make_layers(cfg4, 256)
        self.f5 = self._make_layers(cfg5, 512)
        self.layer = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)
        
        
        if state_dict is not None:
            self.load_state_dict(state_dict)
        
        
        self.x_train = torch.Tensor(train_data)
        self.y_train = torch.LongTensor(train_targets)
        
        
        
        
#         self.hidden_layers = hidden_layers
#         self.aise_params = aise_params
        self.attack_cnn = attack_cnn
    
    
    
        self.deep_knn = CKNN(
            model         = self, 
            device        = DEVICE, 
            x_train = self.x_train,
            y_train = self.y_train,
            batch_size    = 1000,
            n_neighbors   = 10,
            n_embs        = 4 
            )

    def truncated_forward(self,truncate=None):
        assert truncate is not None,"truncate must be specified"
        if truncate == 0:
            return self.forward1
        elif truncate == 1:
            return self.forward2
        elif truncate == 2:
            return self.forward3
        else:
            return self.forward4

    def forward1(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        return out2
    
    def forward2(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        return out3
    
    def forward3(self, x):
#         out1 = self.f1(x)
#         out2 = self.f2(out1)
        # out3 = self.f3(x)
        out4 = self.f4(x)
        return out4
    
    def forward4(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out45 = self.f5(out4)
        out5 = self.layer(out45)
        return out5
    
    
    def feature_forward(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out45 = self.f5(out4)
        out5 = self.layer(out45)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return out2, out3, out4, out5, out
    
    

    def forward(self, x):
        #out = self.features(x)
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out45 = self.f5(out4)
        out5 = self.layer(out45)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return [out,]

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



    def __call__(self,x):
        # check if channel first
        if x.size(1) != 3:
            x = x.permute([0,3,1,2])
        return self.forward(x)

    def predict(self, x):
        # check if channel first
        if x.size(1) != 3:
            x = x.permute([0,3,1,2])
        
        pred,_,_ = self.deep_knn.predict(x)
        print(pred)
        return pred

        # pred_sum = 0.
        # for i in range(len(self.hidden_layers)):
        #     pred_sum = pred_sum + self.aise[i](x)
        # return pred_sum / len(self.hidden_layers)

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


def make_cifar_model(**kwargs):
    return VGG(**kwargs)


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    assert weights_path is not None
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model = make_cifar_model(train_data=checkpoint["train_data"],train_targets=checkpoint["train_targets"],
                             state_dict=checkpoint["state_dict"])
    model.to(DEVICE)
    #model.load_state_dict(checkpoint["state_dict"])
    
    for params in model.parameters():
        params.requires_grad_(False)

    wrapped_model = RAILSEvalWrapper(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
