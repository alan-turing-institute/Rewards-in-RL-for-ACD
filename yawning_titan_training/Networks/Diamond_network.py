from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
from datetime import datetime


class GenerateDiamondNetwork:
    # Define static UUIDs for the network and a base UUID for nodes (you might want to replace this with a UUID generator)
    STATIC_NETWORK_UUID = "122d031c-c8ba-430e-a32b-9d0fb5db1975"  # Network UUID
    BASE_NODE_UUID = "00000000-0000-0000-0000-00000000000"  # Base UUID for nodes (you can modify this as needed)

    def __init__(self, num_nodes=4):
        """
        Note: `num_nodes` is not being used but kept in case I need it in the future
        """
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

        # Create 4 nodes in a diamond shape
        node1 = Node("host_1")
        node1._uuid = self.BASE_NODE_UUID + "01"
        node1.node_position = [-2, 2]  # Left entry node
        node1.vulnerability = 1

        node2 = Node("host_2")
        node2._uuid = self.BASE_NODE_UUID + "02"
        node2.node_position = [0, 4]  # top
        node2.vulnerability = 0.5

        node3 = Node("host_3")
        node3._uuid = self.BASE_NODE_UUID + "03"
        node3.node_position = [0, 0]  # bottom
        node3.vulnerability = 1

        node4 = Node("host_4")
        node4._uuid = self.BASE_NODE_UUID + "04"
        node4.node_position = [2, 2]  # far right
        node4.vulnerability = 1

        # Add nodes to the network
        network.add_node(node1)
        network.add_node(node2)
        network.add_node(node3)
        network.add_node(node4)

        # Connect edges to form a diamond
        network.add_edge(node1, node2)
        network.add_edge(node1, node3)
        network.add_edge(node2, node4)
        network.add_edge(node3, node4)

        # Mark the left node as the entry node
        node1.entry_node = True

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