from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
from datetime import datetime


class GenerateSubproblemNetwork:
    # Define static UUIDs for the network and a base UUID for nodes (you might want to replace this with a UUID generator)
    STATIC_NETWORK_UUID = "122d031c-c8ba-430e-a32b-9d0fb5db1975"  # Network UUID
    BASE_NODE_UUID = "00000000-0000-0000-0000-00000000000"  # Base UUID for nodes (you can modify this as needed)

    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        self.network = self.build_network()

    def build_network(self):
        # Instantiate network
        network = Network(
            set_random_entry_nodes=False,
            num_of_random_entry_nodes=0,
            set_random_high_value_nodes=False,
            num_of_random_high_value_nodes=0,
            set_random_vulnerabilities=False,
        )

        nodes = []
        for i in range(1, self.num_nodes + 1):
            # Create each node with a unique UUID and name
            node_name = f"host_{i}"
            node_uuid = f"{self.BASE_NODE_UUID}{i:02d}"  # Creating a unique UUID by appending the node number
            node = Node(node_name)
            node._uuid = node_uuid
            node.node_position = [4.0 * (i - 1), 2.0]  # Position nodes laterally with a gap of 4 units
            node.vulnerability = 1
            nodes.append(node)

            # Add node to the network
            network.add_node(node)

            # Connect this node to the previous node if it exists
            if i > 1:
                network.add_edge(nodes[i - 2], node)

        # Set the first node as the entry node
        nodes[0].entry_node = True

        # Convert the network to the required dictionary format
        network_dict = self.network_to_dict(network)

        return network_dict

    def network_to_dict(self, network):
        # Create the nodes dictionary with static UUIDs
        nodes_dict = {}
        for node in network.nodes:
            nodes_dict[node.uuid] = {
                "uuid": node.uuid,
                "name": node.name,
                "high_value_node": node.high_value_node,
                "entry_node": node.entry_node,
                "vulnerability": node.vulnerability,
                "x_pos": node.node_position[0],
                "y_pos": node.node_position[1],
            }

        # Create the edges dictionary
        edges_dict = {}
        for node in network.nodes:
            connected_edges = {}
            for edge in network.edges:
                if edge[0] == node:
                    connected_edges[edge[1].uuid] = {}
                elif edge[1] == node:
                    connected_edges[edge[0].uuid] = {}
            edges_dict[node.uuid] = connected_edges

        # Compile the full network dictionary
        network_dict = {
            "set_random_entry_nodes": network.set_random_entry_nodes,
            "random_entry_node_preference": "NONE",
            "num_of_random_entry_nodes": network.num_of_random_entry_nodes,
            "set_random_high_value_nodes": network.set_random_high_value_nodes,
            "random_high_value_node_preference": "NONE",
            "num_of_random_high_value_nodes": network.num_of_random_high_value_nodes,
            "set_random_vulnerabilities": network.set_random_vulnerabilities,
            "node_vulnerability_lower_bound": 0.01,
            "node_vulnerability_upper_bound": 1,
            "nodes": nodes_dict,
            "edges": edges_dict,
            "_doc_metadata": {
                "uuid": self.STATIC_NETWORK_UUID,  # Use the static UUID for the network
                "locked": False,
                "created_at": datetime.now().isoformat()
            }
        }

        return network_dict
