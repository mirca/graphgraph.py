import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolutionalLayer(Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = Parameter(torch.DoubleTensor(self.input_dim, self.output_dim))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(self.output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        u = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-u, u)
        if self.bias is not None:
            self.bias.data.uniform_(-u, u)

    def forward(self, hidden_features, adjacency_matrix):
        """
        return AUW+b for the GNN
        """

        embeddings = torch.mm(adjacency_matrix, torch.mm(hidden_features, self.W))
        if self.bias is not None:
            return embeddings + self.bias
        else:
            return embeddings

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.input_dim)
            + " -> "
            + str(self.output_dim)
            + ")"
        )


class GraphConvolutionalNeuralNet(Module):
    """
    a GCN model with multiple layers
    """

    def __init__(self, node_features_dim, hidden_dim, output_dim, bias=True):
        super(GraphConvolutionalNeuralNet, self).__init__()
        self.layer1 = GraphConvolutionalLayer(node_features_dim, hidden_dim, bias=bias)
        self.layer2 = GraphConvolutionalLayer(hidden_dim, output_dim, bias=bias)

    def forward(self, node_features, adjacency_matrix):
        self.embeddings_1 = torch.nn.functional.relu(
            self.layer1(node_features, adjacency_matrix)
        )
        self.embeddings_2 = self.layer2(self.embeddings_1, adjacency_matrix)
        output = torch.nn.functional.log_softmax(self.embeddings_2, dim=1)
        return output


class SimpleGraphConvolutionalNeuralNet(Module):
    """
    a GCN model with multiple layers
    """

    def __init__(self, node_features_dim, hidden_dim, output_dim, bias=True):
        super(GraphConvolutionalNeuralNet, self).__init__()
        self.layer1 = GraphConvolutionalLayer(node_features_dim, hidden_dim, bias=bias)
        self.layer2 = GraphConvolutionalLayer(hidden_dim, output_dim, bias=bias)

    def forward(self, node_features, adjacency_matrix):
        self.embeddings_1 = self.layer1(node_features, adjacency_matrix)
        self.embeddings_2 = self.layer2(self.embeddings_1, adjacency_matrix)
        output = torch.nn.functional.log_softmax(self.embeddings_2, dim=1)
        return output


"""
model = GraphConvolutionalNeuralNet(node_features.shape[1], 128, 2)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.NLLLoss()

loss_at_epoch = np.zeros(1000)
for i in tqdm(range(1000)):
    net = model(node_features, adjacency_matrix)
    loss = criterion(net, labels)
    loss_at_epoch[i] = loss.item()
    loss.backward()
    optimizer.step()

torch.exp(model(node_features, adjacency_matrix))
"""
