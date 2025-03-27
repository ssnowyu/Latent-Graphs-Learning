from typing import Any

from torch_geometric.data import Data

# class SnapshotData(Data):
#     """
#     x: (num_nodes, dim_node_feature)
#     temporal_edge_index: (2, num_edge)
#     step_index: (1, num_edge)
#     target_edge_index: (2, num_edge)
#     """
#     def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == 'temporal_edge_index':
#             return self.x.size(0)
#         if key == 'target_edge_index':
#             return self.x.size(0)
#         if key == 'step_index':
#             return 0
#         return Data.__inc__(self, key, value, *args, **kwargs)
#
#     def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == 'temporal_edge_index':
#             return 1
#         if key == 'target_edge_index':
#             return 1
#         if key == 'step_index':
#             return 1
#         return Data.__cat_dim__(self, key, value, *args, **kwargs)


class SnapshotData(Data):
    """
    x: (num_nodes, dim_node_feature)
    temporal_edge_index: (2, num_edge)
    step_index: (1, num_edge)
    target_edge_index: (2, num_edge)
    """

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "temporal_edge_index":
            return self.x.size(0)
        if key == "target_edge_index":
            return self.x.size(0)
        if key == "step_index":
            return 0
        return Data.__inc__(self, key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "temporal_edge_index":
            return 1
        if key == "target_edge_index":
            return 1
        if key == "step_index":
            return 1
        return Data.__cat_dim__(self, key, value, *args, **kwargs)
