import random
def display_dataset_discription(dataset):    
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print("number of node features:\t",dataset.num_node_features)
    print(f'# classes/unique regression value: {dataset.num_classes}')


    random_index = random.randrange(len(dataset))  
    sample_data = dataset[random_index]  # Get the first graph object.

    print(f"Get a random graph object: Index {random_index}")
    print(sample_data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {sample_data.num_nodes}')
    print(f'Number of edges: {sample_data.num_edges}')
    print(f'Average node degree: {sample_data.num_edges / sample_data.num_nodes:.2f}')
    print(f'Has isolated nodes: {sample_data.has_isolated_nodes()}')
    print(f'Has self-loops: {sample_data.has_self_loops()}')
    print(f'Is undirected: {sample_data.is_undirected()}')
