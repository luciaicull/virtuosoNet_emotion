import numpy as np
import torch as th



def edges_to_matrix(edges, num_notes, model_config):
    if not model_config.is_graph:
        return None
    num_keywords = len(model_config.graph_keys)
    matrix = np.zeros((model_config.num_edge_types, num_notes, num_notes))

    for edg in edges:
        if edg[2] not in model_config.graph_keys:
            continue
        edge_type = model_config.graph_keys.index(edg[2])
        matrix[edge_type, edg[0], edg[1]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1], edg[0]] = 1
        else:
            matrix[edge_type, edg[1], edg[0]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)
    return matrix


def edges_to_matrix_short(edges, slice_index, model_config):
    if not model_config.is_graph:
        return None
    num_keywords = len(model_config.graph_keys)
    num_notes = slice_index[1] - slice_index[0]
    matrix = np.zeros((model_config.num_edge_types, num_notes, num_notes))
    start_edge_index = binary_index_for_edge(
        edges, slice_index[0])
    end_edge_index = binary_index_for_edge(
        edges, slice_index[1] + 1)
    for i in range(start_edge_index, end_edge_index):
        edg = edges[i]
        if edg[2] not in model_config.graph_keys:
            continue
        if edg[1] >= slice_index[1]:
            continue
        edge_type = model_config.graph_keys.index(edg[2])
        matrix[edge_type, edg[0]-slice_index[0], edg[1]-slice_index[0]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1] -
                   slice_index[0], edg[0]-slice_index[0]] = 1
        else:
            matrix[edge_type, edg[1]-slice_index[0], edg[0]-slice_index[0]] = 1
    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)

    return matrix


def edges_to_sparse_tensor(edges, model_config):
    num_keywords = len(model_config.graph_keys)
    edge_list = []
    edge_type_list = []

    for edg in edges:
        edge_type = model_config.graph_keys.index(edg[2])
        edge_list.append(edg[0:2])
        edge_list.append([edg[1], edg[0]])
        edge_type_list.append(edge_type)
        if edge_type != 0:
            edge_type_list.append(edge_type+num_keywords)
        else:
            edge_type_list.append(edge_type)

        edge_list = th.LongTensor(edge_list)
    edge_type_list = th.FloatTensor(edge_type_list)

    matrix = th.sparse.FloatTensor(edge_list.t(), edge_type_list)

    return matrix


def binary_index_for_edge(alist, item):
    first = 0
    last = len(alist) - 1
    midpoint = 0

    if (item < alist[first][0]):
        return 0

    while first < last:
        midpoint = (first + last) // 2
        currentElement = alist[midpoint][0]

        if currentElement < item:
            if alist[midpoint + 1][0] > item:
                return midpoint
            else:
                first = midpoint + 1
            if first == last and alist[last][0] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint - 1
        else:
            if midpoint + 1 == len(alist):
                return midpoint
            while midpoint >= 1 and alist[midpoint - 1][0] == item:
                midpoint -= 1
                if midpoint == 0:
                    return midpoint
            return midpoint
    return last