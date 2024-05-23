import pandas as pd
import torch
from torch_geometric.data import Data

def load_data(file_path):
    """
    Load data from a text file and return as a Pandas DataFrame.
    Args:
        file_path (str): The path to the text file contaning the data.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    # Assuming the file is a comma-sperated values (csv) or txt file

    try:
        # Read the file using pandas read_csv, assuming it's comma-separated
        df = pd.read_csv(file_path, delimiter=',')
        print(f"Data loaded successfully from {file_path}")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_graph_data(df):
    """
    Create graph data from the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        Data: A PyTorch Geometric Data object containing the graph data.
    """

    # Create node features for customers and products
    customers = df['customer'].unique()
    products = df['product'].unique()

    customer_index = {customer: idx for idx, customer in enumerate(customers)}
    product_index = {product: idx + len(customers) for idx, product in enumerate(products)}

    # Create edge index for customer-product relationsships
    edges = []
    for _, row in df.iterrows():
        customer = row['customer']
        product = row['product']
        edges.append([customer_index[customer], product_index[product]])
        edges.append([product_index[product], customer_index[customer]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create node features
    num_nodes = len(customers) + len(products)
    node_features = torch.zeros((num_nodes, 6), dtype=torch.float)
    for _, row in df.iterrows():
        customer = row['customer']
        product = row['product']
        customer_idx = customer_index[customer]
        product_idx = product_index[product]
        node_features[customer_idx] = torch.tensor([row['order'], row['product'], row['customer'], 0, 0, 0], dtype=torch.float)
        node_features[product_idx] = torch.tensor([row['order'], row['product'], row['customer'], row['color'], row['size'], row['group']], dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index)

'''
# Example usage
if __name__ == "__main__":
    file_path = 'order_data.txt'
    df = load_data(file_path)
    if df is not None:
        print(df.head())
'''