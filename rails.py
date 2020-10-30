from art.classifiers import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from AISE import AISE
import pickle,json
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RAILSEvalWrapper(PyTorchClassifier):
    def __init__(self,**kwargs):
        super(RAILSEvalWrapper,self).__init__(**kwargs)

    def predict(self, x, batch_size=128, **kwargs):

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            with torch.no_grad():
                output = self._model.predict(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            results[begin:end] = output.detach().cpu().numpy()

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions


class CNNAISE(nn.Module):
    def __init__(self, train_data, train_targets, hidden_layers, aise_params):
        super(CNNAISE, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.x_train = train_data
        self.y_train = train_targets
        self.hidden_layers = hidden_layers
        self.aise_params = aise_params

    def truncated_forward(self, truncate=None):
        assert truncate is not None, "truncate must be specified"
        if truncate == 0:
            return self.partial_forward_1
        elif truncate == 1:
            return self.partial_forward_2
        elif truncate == 2:
            return self.partial_forward_3
        else:
            return self.partial_forward_4

    def partial_forward_1(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training=self.training)
        return out_conv1

    def partial_forward_2(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training=self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training=self.training)
        return out_conv2

    def partial_forward_3(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training=self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training=self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size=(2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training=self.training)
        return out_conv3

    def partial_forward_4(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training=self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training=self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size=(2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training=self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training=self.training)
        return out_conv4

    def forward(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training=self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training=self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size=(2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training=self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training=self.training)
        out_pool2 = F.max_pool2d(out_conv4, kernel_size=(2, 2))
        out_view = out_pool2.view(-1, 128 * 7 * 7)
        out_fc1 = F.dropout(F.relu(self.fc1(out_view)), 0.1, training=self.training)
        out_fc2 = F.dropout(F.relu(self.fc2(out_fc1)), 0.1, training=self.training)
        out = self.fc3(out_fc2)

        return out

    def predict(self, x):
        pred_sum = 0.
        for i, layer in enumerate(self.hidden_layers):
            aise = AISE(self.x_train, self.y_train, hidden_layer=layer, model=self, **self.aise_params[str(i)])
            pred_sum = pred_sum + aise(x)
        return pred_sum / len(self.hidden_layers)


def make_mnist_model(**kwargs):
    return CNNAISE(**kwargs)


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    assert weights_path is not None
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model = make_mnist_model(train_data=checkpoint["train_data"],train_targets=checkpoint["train_targets"],**model_kwargs)
    model.to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    wrapped_model = RAILSEvalWrapper(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
