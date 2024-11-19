REGISTRY = {}
from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .gnn_agent import GNNAgent
from .pyg_gnn_agent import GNNAgent as PyGGNNAgent
from .off_rnn_agent import OffRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["pyggnn"] = PyGGNNAgent
REGISTRY["off_rnn"] = OffRNNAgent
