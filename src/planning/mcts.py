import torch
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from src.env.action_space import ActionSpace
from src.utils.efe import compute_EFE_igpv
from src.utils.utils import attach_goal_mask, generate_goal_mask
import time

class Node:
    def __init__(self, action_space:ActionSpace, stochastic_state=None, deterministic_state=None, parent=None, depth=0, preceeding_action=None, prior_prob=0.0):
        self.s = stochastic_state
        self.h = deterministic_state
        self.action_space = action_space
        self.parent = parent
        self.depth = depth
        self.preceeding_action = preceeding_action
        self.children = {} # action -> Node
        self.efe = 0.0
        self.visits = 0
        self.ucb = 0
        self.prior_prob = prior_prob
        self.ig = 0.0
        self.pv = 0.0

        self.point_efe = 0.
        self.point_ig = 0.
        self.point_pv = 0.

        self.child_priors = np.zeros(self.action_space.total_size())
        self.child_efes = np.zeros(self.action_space.total_size())
        self.child_visits = np.zeros(self.action_space.total_size())
        self.child_probs = np.zeros(self.action_space.total_size())
        self.child_ucbs = np.zeros(self.action_space.total_size())

        self.child_ig = np.zeros(self.action_space.total_size())
        self.child_pv = np.zeros(self.action_space.total_size())

        self.child_point_efes = np.zeros(self.action_space.total_size())
        self.child_point_ig = np.zeros(self.action_space.total_size())
        self.child_point_pv = np.zeros(self.action_space.total_size())

    def add_child(self, action, child):
        self.children[action] = child
        if child is not None:
            child.parent = self
        if child.preceeding_action is None:
            child.preceeding_action = action

    def expand(self):
        for action in self.action_space.actions():
            if action not in self.children:
                self.add_child(action, Node(self.action_space, parent=self, depth=self.depth+1))
        return self.children
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def is_fully_expanded(self):
        return len(self.children) == self.action_space.total_size()

    def select_action(self, greedy=False):
        if greedy:
            return np.argmax(self.child_priors)
        return self.action_space.sample(probs=self.child_probs)

    def compute_ucb(self, c=1.0, use_priors=False, original_ucb=False):
        #self.child_ucbs = - (self.child_efes / self.child_visits) + c * self.child_priors * np.sqrt(self.visits) / (1 + self.child_visits)
        self.child_ucbs = - self.get_normalized_efes() + self.get_exploration_bonus(c=c, use_priors=use_priors, original_ucb=original_ucb)

    def update_child_visits(self):
        self.child_visits = np.array([child.visits for child in self.children.values()])

    def update_child_efes(self):
        self.child_efes = np.array([child.efe for child in self.children.values()])

    def update_child_priors(self):
        self.child_priors = np.array([child.prior_prob for child in self.children.values()])

    def update_child_ig(self):
        self.child_ig = np.array([child.ig for child in self.children.values()]) 
    
    def update_child_pv(self):
        self.child_pv = np.array([child.pv for child in self.children.values()])

    def get_normalized_efes(self):
        # Avoid division by zero by setting zero visits to have zero EFE
        return np.divide(self.child_efes, self.child_visits, out=np.zeros_like(self.child_efes), where=self.child_visits!=0)
    
    def get_exploration_bonus(self, c=1.0, use_priors=False, original_ucb=False):
        if original_ucb:
            if use_priors:
                return c * self.child_priors * np.sqrt(self.visits) / (1 + self.child_visits)
            else:
                return c * np.sqrt(self.visits) / (1 + self.child_visits)
        else:
            if use_priors:
                return c * self.child_priors * 1 / (1 + self.child_visits)
            else:
                return c * 1 / (1 + self.child_visits)

    def update_child_probs(self, temperature=1.0):
        # Apply softmax to convert UCBs to probabilities
        # Temperature parameter controls exploration (high temp) vs exploitation (low temp)
        # Subtracting max for numerical stability
        ucb_max = np.max(self.child_ucbs)
        exp_ucb = np.exp((self.child_ucbs - ucb_max) / temperature)
        sum_exp_ucb = np.sum(exp_ucb)
        if sum_exp_ucb > 0:
            self.child_probs = exp_ucb / sum_exp_ucb
        else:
            # If all UCBs are extremely negative, use uniform distribution
            self.child_probs = np.ones_like(self.child_ucbs) / len(self.child_ucbs)
    
    def final_action_probs(self, use_efe=False):
        if use_efe:
            # Apply softmax to negative average EFEs (lower EFE is better)
            avg_efes = self.child_efes / np.maximum(self.child_visits, 1)  # avoid division by zero
            neg_avg_efes = -avg_efes
            # For numerical stability, subtract max
            neg_avg_efes_shifted = neg_avg_efes - np.max(neg_avg_efes)
            exp_neg_avg_efes = np.exp(neg_avg_efes_shifted)
            return exp_neg_avg_efes / np.sum(exp_neg_avg_efes)
        
        visits_plus_one = self.child_visits + 1
        # Check for zero sum to avoid division by zero
        visits_sum = np.sum(visits_plus_one)
        if visits_sum > 0:
            return visits_plus_one / visits_sum
        else:
            # Return uniform distribution if no visits
            return np.ones_like(self.child_visits) / len(self.child_visits)

# MCTS Steps:
# 1. Selection: Start at the root node and select successive child nodes until a leaf node is reached.
# 2. Expansion: Unless the leaf node ends the game, expand it by adding one (or more) child nodes.
# 3. Simulation: Run a simulation from the selected node until a result is achieved.
# 4. Backpropagation: Use the result of the simulated playout to update information in the nodes on the path from the root to the leaf.
class MCTSTree:
    def __init__(self, root:Node, action_space:ActionSpace, preference, 
                 gamma=0.99, 
                 c=1.0, 
                 greedy=False, 
                 max_depth=10, 
                 decision_threshold=0.5, 
                 recursive_update=False, 
                 goal_precision=1.0, 
                 select_with_efe=False,
                 compute_mask_from_recon=False,
                 max_rollout_depth=40,
                 fixed_rollout_depth=False,
                 verbose=False):
        self.root = root
        self.action_space = action_space
        self.gamma = gamma
        self.c = c
        self.greedy = greedy
        self.preference = preference
        self.max_depth = max_depth
        self.max_rollout_depth = max_rollout_depth
        self.fixed_rollout_depth = fixed_rollout_depth
        self.decision_threshold = decision_threshold
        self.recursive_update = recursive_update
        self.goal_precision = goal_precision
        self.select_with_efe = select_with_efe
        self.compute_mask_from_recon = compute_mask_from_recon
        self.verbose = verbose

    def search(self, model, policy, num_simulations=100, alt_backprop=False, discounting=False, num_rollouts=10):
        iter_times = []
        # TODO: Quit early if habitual prior is sharp enough
        tick = time.perf_counter()

        for i in range(num_simulations):
            if self.verbose:
                print(f"Beginning iteration {i+1}/{num_simulations}")
            iter_start = time.perf_counter()
            if self.decision_threshold_reached(self.root.final_action_probs()):
                print("Decision threshold reached, quitting early")
                print(f"Final action probs: {self.root.final_action_probs()}")
                break
            node = self.select(self.root)
            if node.depth != self.max_depth:
                new_nodes = self.expand(node, model, policy)
                if self.fixed_rollout_depth:
                    # Use the same depth for all nodes
                    depth = self.max_rollout_depth - node.depth
                else:
                    depth = self.max_rollout_depth 
            efe, ig, pv = self.rollout(node, model, policy, depth=depth, discount=discounting, num_samples=num_rollouts)
            if alt_backprop:
                self.backpropagate_alt(node, efe.mean(), ig.mean(), pv.mean())
            else:
                self.backpropagate(node, efe.mean(), ig.mean(), pv.mean())
            iter_end = time.perf_counter()
            iter_times.append(iter_end - iter_start)

        tock = time.perf_counter()
        print(f"Search took {tock - tick:.2f} seconds")
        print(f"Average iteration time: {np.mean(iter_times):.4f} seconds")
        try:
            final_action,_ = self.action_space.sample(probs=self.root.final_action_probs(use_efe=self.select_with_efe))
        except:
            final_action = 3
            print("Action probs probably contain NaN values, defaulting to action 3")

        best_action_path = self.get_best_path(self.root, max_depth=self.max_depth, use_efe=self.select_with_efe)

        print(f"Best action path: {best_action_path}")
        # Print final statistics
        if self.verbose:
            print("\n=== Final Tree Statistics ===")
            print(f"Total root visits: {self.root.visits}")
            try:
                print(f"Action probabilities: {self.root.final_action_probs(use_efe=self.select_with_efe)}")
            except:
                print("Action probabilities probably contain NaN values")
            print(f"Child visits: {self.root.child_visits}")
            print(f"Child EFEs: {self.root.child_efes / self.root.child_visits}")
            print(f"Child priors: {self.root.child_priors}")
            print(f"Child UCBs: {self.root.child_ucbs}")
            print(f"Child IG: {self.root.child_ig / self.root.child_visits}")
            print(f"Child PV: {self.root.child_pv / self.root.child_visits}")
            print(f"Child Point EFEs: {self.root.child_point_efes}")
            print(f"Child Point IG: {self.root.child_point_ig}")
            print(f"Child Point PV: {self.root.child_point_pv}")
            print(f"Child Normalized EFEs: {self.root.get_normalized_efes()}")
            print(f"Child Exploration Bonuses: {self.root.get_exploration_bonus()}")
            print(f"Max depth: {self.get_max_depth()}")
            print(f"Final action: {final_action}")
            print("=============================")
        return final_action, best_action_path
        #return self.root.select_action(self.greedy)

    def select(self, node:Node):
        while not node.is_leaf():
            action,_ = node.select_action()
            try:
                node = node.children[action]
                if self.verbose:
                    print(f"Selected action: {action}, node depth: {node.depth}")
            except:
                raise ValueError("Action not in children")
        return node
    
    def rollout(self, node:Node, model, policy, depth=10, discount=False, relative_discount=False, num_samples=10):
        if self.verbose:
            print(f"Rollout from depth: {node.depth}")
        z_tensor, h_tensor, dist_tensor, x_hat, x_hat_mask, dist_from_recon = model.rollout_policy_network(node.s, policy, node.h, depth=depth, num_samples=num_samples, recon=True)
        
        if self.compute_mask_from_recon:
            x_hat_mask = generate_goal_mask(x_hat)
        
        posterior = D.OneHotCategorical(logits=dist_tensor[0])
        posterior_given_o = D.OneHotCategorical(logits=dist_from_recon[0])

        # Compute the expected free energy
        efe, ig, pv = compute_EFE_igpv(posterior, posterior_given_o, x_hat_mask, self.preference, goal_is_bernoilli=True, goal_precision=self.goal_precision, average_over_pixels=True, reduce=False)

        if discount:
            if relative_discount:
                # discount is relative to start node and not absolute time-step
                start_depth = 1
            else:
                start_depth = node.depth + 1 
            # Discount the expected free energy based on the depth of the node
            discounts = torch.pow(self.gamma, torch.arange(start_depth, start_depth + depth).to(node.s.device))
            # Reshape discounts for proper broadcasting along time dimension (dim 2)
            # For (batch x samples x timesteps), we need (1, 1, timesteps)
            discounts = discounts.view(1, 1, -1)

            # Apply discounts to each metric - sum across timesteps, mean across samples
            efe = (efe * discounts).sum(dim=2).mean(dim=1)
            ig = (ig * discounts).sum(dim=2).mean(dim=1)
            pv = (pv * discounts).sum(dim=2).mean(dim=1)
        else:
            efe = efe.sum(dim=2).mean(dim=1)
            ig = ig.sum(dim=2).mean(dim=1)
            pv = pv.sum(dim=2).mean(dim=1)

        return efe, ig, pv

    def expand(self, node:Node, model, policy):
        # when expanding we need to:
        # 1. Transition the state using the transition model
        # 2. Get action a priori probabilities using the habitual network
        # 3. Estimate expected free energy through model rollouts under the habitual policy
        # 4. Update the node with the expected free energy and prior probabilities
        #print(f"Expanding node at depth {node.depth}")
        new_nodes = node.expand()
        _,prior_probs,_ = policy(node.s, node.h)
        prior_probs = prior_probs.squeeze(0)
        # Handle different tensor shapes based on dimensionality
        if len(node.s.shape) == 3:  # (batch, num_dims, dim_classes)
            s_repeated = node.s.repeat(self.action_space.total_size(), 1, 1)
        else:
            s_repeated = node.s.repeat(self.action_space.total_size(), 1)
            
        h_repeated = node.h.repeat(self.action_space.total_size(), 1)
        
        all_actions = self.action_space.all_actions_as_tensor(device=node.s.device)
        s_next, dist_next, h_next = model.transition(s_repeated, all_actions, h_repeated)
        
        x_hat_next, x_hat_mask_next = model.decode(s_next, h_next)
        if self.compute_mask_from_recon:
            x_hat_mask_next = generate_goal_mask(x_hat_next)
            
        _, dist_from_recon = model.encode(attach_goal_mask(x_hat_next, x_hat_mask_next), h_next)
        posterior = D.OneHotCategorical(logits=dist_next[0])
        posterior_given_o = D.OneHotCategorical(logits=dist_from_recon[0])

        # Compute the expected free energy
        efe, ig, pv = compute_EFE_igpv(posterior, posterior_given_o, x_hat_mask_next, self.preference, goal_is_bernoilli=True, goal_precision=self.goal_precision, reduce=False, average_over_pixels=True)

        for action, child in new_nodes.items():
            if child is not None:
                child.s = s_next[action].unsqueeze(0)
                child.visits = 0
                child.h = h_next[action].unsqueeze(0)
                child.prior_prob = prior_probs[action].unsqueeze(0)
                # child.efe = efe[action]
                # child.ig = ig[action]
                # child.pv = pv[action]
                child.point_efe = efe[action]
                child.point_ig = ig[action]
                child.point_pv = pv[action]
        node.update_child_visits()
        
        #node.update_child_efes()
        node.child_priors = prior_probs.cpu().numpy()
        #node.child_efes = efe.cpu().numpy()
        #node.child_ig = ig.cpu().numpy()
        #node.child_pv = pv.cpu().numpy()

        node.child_point_efes = efe.cpu().numpy()
        node.child_point_ig = ig.cpu().numpy()
        node.child_point_pv = pv.cpu().numpy()

        node.compute_ucb(self.c)
        node.update_child_probs()
        return new_nodes

    def backpropagate(self, node:Node, efe, ig, pv):
        while node is not None:
            node.visits += 1
            efe = node.point_efe + self.gamma * efe
            ig = node.point_ig + self.gamma * ig
            pv = node.point_pv + self.gamma * pv
            node.efe += efe
            node.ig += ig
            node.pv += pv
            node.compute_ucb(self.c)
            node.update_child_probs()
            if node.is_root():
                break
            node.parent.child_efes[node.preceeding_action] += efe
            node.parent.child_ig[node.preceeding_action] += ig
            node.parent.child_pv[node.preceeding_action] += pv
            node.parent.child_visits[node.preceeding_action] += 1
            # if self.recursive_update:
            #     efe = node.efe
            #     ig = node.ig
            #     pv = node.pv
            node = node.parent

    def backpropagate_alt(self, node:Node, efe, ig, pv, act_style=True):
        while node is not None:
            node.visits += 1
            if act_style:
                node.efe = node.efe + (efe - node.efe) / node.visits
                node.ig = node.ig + (ig - node.ig) / node.visits
                node.pv = node.pv + (pv - node.pv) / node.visits
            else:
                node.efe = node.point_efe + (efe - node.point_efe) / node.visits
                node.ig = node.point_ig + (ig - node.point_ig) / node.visits
                node.pv = node.point_pv + (pv - node.point_pv) / node.visits
            #efe = node.efe
            #ig = node.ig
            #pv = node.pv
            node.compute_ucb(self.c)
            node.update_child_probs()
            if node.is_root():
                break
            node.parent.child_efes[node.preceeding_action] = node.efe
            node.parent.child_ig[node.preceeding_action] = node.ig
            node.parent.child_pv[node.preceeding_action] = node.pv
            node.parent.child_visits[node.preceeding_action] += 1
            node = node.parent

    def get_max_depth(self):
        return self.get_max_depth_recursive(self.root)
    
    def get_max_depth_recursive(self, node:Node):
        if node.is_leaf():
            return 0
        return 1 + max(self.get_max_depth_recursive(child) for child in node.children.values())

    # Extract the sequence of actions along the best path
    def get_best_path(self, node, max_depth=5, use_efe=False):
        path = []
        current = node
        depth = 0
        
        while not current.is_leaf() and depth < max_depth:
            if use_efe:
                # Use negative EFE (lower EFE is better)
                avg_efes = current.child_efes / np.maximum(current.child_visits, 1)
                best_action = int(np.argmin(avg_efes))
            else:
                # Use visit count (higher visits is better)
                best_action = int(np.argmax(current.child_visits))
            
            path.append(best_action)
            if best_action not in current.children:
                break
            current = current.children[best_action]
            depth += 1
        
        return path

    def decision_threshold_reached(self, P, axis=0):
        return (np.max(P, axis=axis) - np.mean(P, axis=axis)) > self.decision_threshold

    def save_tree_data(self, filename):
        """
        Save tree statistics and structure to a JSON file.
        
        Args:
            filename: Output filename (without extension)
        """
        import json
        import os
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Convert node to dictionary
        def node_to_dict(node, depth=0, max_depth=10):
            if node is None or depth > max_depth:
                return None
                
            # Create base node data
            node_data = {
                "depth": depth,
                "visits": int(node.visits),
                "efe": float(node.efe / node.visits) if hasattr(node, 'efe') else None,
                "ig": float(node.ig / node.visits) if hasattr(node, 'ig') else None,
                "pv": float(node.pv / node.visits) if hasattr(node, 'pv') else None,
                "children": {}
            }
            
            # Add action statistics
            if hasattr(node, 'child_probs') and hasattr(node, 'child_visits') and hasattr(node, 'child_efes'):
                actions = []
                for i in range(len(node.child_probs)):
                    actions.append({
                        "action": int(i),
                        "prob": float(node.child_probs[i]),
                        "visits": int(node.child_visits[i]),
                        "efe": float(node.child_efes[i]),
                        "normalized_efe": float(node.child_efes[i] / max(1, node.child_visits[i]))
                    })
                node_data["actions"] = actions
            
            # Add children recursively
            for action, child in node.children.items():
                if child is not None:
                    node_data["children"][str(action)] = node_to_dict(child, depth + 1, max_depth)
                    
            return node_data
        
        # Convert tree to dictionary
        tree_data = {
            "root": node_to_dict(self.root),
            "max_depth": self.max_depth,
            "best_path": self.get_best_path(self.root, max_depth=self.max_depth, use_efe=self.select_with_efe)
        }
        
        # Save as JSON
        with open(f"{filename}.json", 'w') as f:
            json.dump(tree_data, f, indent=2)
            
        return f"{filename}.json"

    def plot_tree(self, filename=None, max_depth=3, highlight_path=None):
        """
        Create a graphical visualization of the MCTS tree.
        
        Args:
            filename: Output filename (without extension)
            max_depth: Maximum depth to visualize (default: 3)
            highlight_path: List of actions to highlight (default: best path)
        
        Returns:
            Graphviz object or filename if saved
        """
        try:
            import graphviz
        except ImportError:
            print("Graphviz package not found. Please install with 'pip install graphviz'")
            return None
        
        # If no highlight path specified, use the best path
        if highlight_path is None:
            highlight_path = self.get_best_path(self.root, max_depth=max_depth, use_efe=self.select_with_efe)
        
        # Create a new directed graph
        dot = graphviz.Digraph(comment='MCTS Tree')
        
        # Configure appearance
        dot.attr('graph', rankdir='TB', size='12,8', dpi='300')
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial')
        dot.attr('edge', fontname='Arial')
        
        # Add nodes and edges recursively
        def add_nodes_to_graph(node, node_id='root', depth=0, path=None):
            if node is None or depth > max_depth:
                return
            
            # Create node label with stats
            visits = node.visits
            efe = node.efe / max(1, visits) if hasattr(node, 'efe') else 0
            ig = node.ig / max(1, visits) if hasattr(node, 'ig') else 0
            pv = node.pv / max(1, visits) if hasattr(node, 'pv') else 0
            
            label = f"Depth {depth}\nVisits: {visits}\nEFE: {efe:.2f}\nIG: {ig:.2f}\nPV: {pv:.2f}"
            
            # Add node to graph
            node_attrs = {'label': label}
            if depth == 0:  # Root node
                node_attrs.update({'fillcolor': 'lightgreen', 'penwidth': '2.0'})
            dot.node(node_id, **node_attrs)
            
            # Add children and edges
            for action, child in node.children.items():
                if child is not None:
                    child_id = f"{node_id}_{action}"
                    
                    # Get action stats
                    action_prob = node.child_probs[action] if hasattr(node, 'child_probs') else 0
                    action_visits = node.child_visits[action] if hasattr(node, 'child_visits') else 0
                    action_efe = node.child_efes[action] / max(1, action_visits) if hasattr(node, 'child_efes') else 0
                    
                    # Edge attributes and label
                    edge_attrs = {}
                    edge_label = f"A{action}\np={action_prob:.2f}\nv={action_visits}\nEFE={action_efe:.2f}"
                    
                    # Highlight edge if on path
                    if path and len(path) > depth and path[depth] == action:
                        edge_attrs.update({'color': 'red', 'penwidth': '2.0'})
                    
                    # Add the edge
                    dot.edge(node_id, child_id, label=edge_label, **edge_attrs)
                    
                    # Recursively add child subtree
                    add_nodes_to_graph(child, child_id, depth + 1, path)
        
        # Build the graph
        add_nodes_to_graph(self.root, path=highlight_path)
        
        # Render and save if filename provided
        if filename:
            try:
                dot.render(filename, view=False, cleanup=True)
                return filename + ".png"
            except Exception as e:
                print(f"Error rendering tree: {e}")
                # Save as DOT file at least
                with open(f"{filename}.dot", "w") as f:
                    f.write(dot.source)
                return f"{filename}.dot"
        
        return dot

    def plot_action_stats(self, filename=None, figsize=(15, 10)):
        """
        Plot action statistics at each depth of the tree.
        
        Args:
            filename: Output filename (optional)
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect stats by depth
        depth_stats = {}
        
        def collect_stats(node, depth=0):
            if node is None:
                return
                
            # Initialize stats for this depth if needed
            if depth not in depth_stats:
                depth_stats[depth] = {
                    'visits': [],
                    'probs': [],
                    'efes': [],
                    'actions': []
                }
            
            # Add stats for this node's actions
            if hasattr(node, 'child_probs') and hasattr(node, 'child_visits') and hasattr(node, 'child_efes'):
                for action in range(len(node.child_probs)):
                    depth_stats[depth]['actions'].append(action)
                    depth_stats[depth]['visits'].append(node.child_visits[action])
                    depth_stats[depth]['probs'].append(node.child_probs[action])
                    depth_stats[depth]['efes'].append(node.child_efes[action] / max(1, node.child_visits[action]))
            
            # Recursively collect stats from children
            for child in node.children.values():
                if child is not None:
                    collect_stats(child, depth + 1)
        
        # Start collection
        collect_stats(self.root)
        
        # Create plots
        max_depth = max(depth_stats.keys()) if depth_stats else 0
        fig, axes = plt.subplots(3, max_depth + 1, figsize=figsize)
        
        # Adjust for the case of only one depth
        if max_depth == 0:
            axes = np.array([axes]).reshape(3, 1)
        
        # Plot rows: visits, probabilities, EFEs
        for depth in range(max_depth + 1):
            if depth in depth_stats:
                stats = depth_stats[depth]
                
                # Group by action
                actions = np.array(stats['actions'])
                unique_actions = np.unique(actions)
                
                visits_by_action = [np.mean([stats['visits'][i] for i in range(len(actions)) if actions[i] == a]) for a in unique_actions]
                probs_by_action = [np.mean([stats['probs'][i] for i in range(len(actions)) if actions[i] == a]) for a in unique_actions]
                efes_by_action = [np.mean([stats['efes'][i] for i in range(len(actions)) if actions[i] == a]) for a in unique_actions]
                
                # Plot visits
                axes[0, depth].bar(unique_actions, visits_by_action)
                axes[0, depth].set_title(f"Depth {depth} Visits")
                axes[0, depth].set_xlabel("Action")
                axes[0, depth].set_ylabel("Visits")
                
                # Plot probabilities
                axes[1, depth].bar(unique_actions, probs_by_action)
                axes[1, depth].set_title(f"Depth {depth} Probabilities")
                axes[1, depth].set_xlabel("Action")
                axes[1, depth].set_ylabel("Probability")
                
                # Plot EFEs
                axes[2, depth].bar(unique_actions, efes_by_action)
                axes[2, depth].set_title(f"Depth {depth} EFEs")
                axes[2, depth].set_xlabel("Action")
                axes[2, depth].set_ylabel("Expected Free Energy")
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        
        return fig

    def visualize_tree(self, max_depth=3):
        """Print a visual representation of the tree."""
        def _print_node(node, depth, max_depth, prefix=""):
            if depth > max_depth:
                return
            
            # Print current node
            visits_str = f"visits={node.visits}"
            efe_str = f"efe={node.efe/node.visits:.4f}" if hasattr(node, 'efe') else ""
            print(f"{prefix}├── Depth {depth}: {visits_str}, {efe_str}")
            
            # Print children
            if not node.children:
                return
                
            sorted_actions = sorted(node.children.keys())
            for i, action in enumerate(sorted_actions):
                child = node.children[action]
                if child is not None:
                    new_prefix = prefix + ("│   " if i < len(sorted_actions) - 1 else "    ")
                    action_prob = node.child_probs[action] if hasattr(node, 'child_probs') else "?"
                    print(f"{prefix}│   Action {action} (p={action_prob:.4f})")
                    _print_node(child, depth + 1, max_depth, new_prefix)
        
        print("\n=== MCTS Tree Visualization ===")
        _print_node(self.root, 0, max_depth)

class OneStepPlanner:
    def __init__(self, action_space:ActionSpace, preference, goal_precision=1.):
        self.action_space = action_space
        #self.model = model
        self.preference = preference
        self.goal_precision = goal_precision

    def plan(self, model, s, h, num_simulations=100, greedy=False):
        if len(s.shape) == 3:  # (batch, num_dims, dim_classes)
            s_repeated = s.repeat(self.action_space.total_size(), 1, 1)
        else:
            s_repeated = s.repeat(self.action_space.total_size(), 1)
            
        h_repeated = h.repeat(self.action_space.total_size(), 1)
        
        all_actions = self.action_space.all_actions_as_tensor(device=s.device)
        s_next, dist_next, h_next = model.transition(s_repeated, all_actions, h_repeated)
        

        x_hat_next, x_hat_mask_next = model.decode(s_next, h_next)
        _, dist_from_recon = model.encode(attach_goal_mask(x_hat_next, x_hat_mask_next), h_next)
        posterior = D.OneHotCategorical(logits=dist_next[0])
        posterior_given_o = D.OneHotCategorical(logits=dist_from_recon[0])

        # Compute the expected free energy
        efe, ig, pv = compute_EFE_igpv(posterior, posterior_given_o, x_hat_mask_next, self.preference, goal_is_bernoilli=True, goal_precision=self.goal_precision, reduce=False)
        efe_np = efe.cpu().numpy()

        print("================================")
        print(f"Expected free energy: {efe_np}")
        print(f"IG: {ig.cpu().numpy()}")
        print(f"PV: {pv.cpu().numpy()}")
        print("================================")

        if greedy:
            return np.argmax(efe_np)
        try:
            action,_ = self.action_space.sample(probs=efe_np)
        except:
            action = 3
        return action

    def get_efe_data(self, model, s, h):
        if len(s.shape) == 3:  # (batch, num_dims, dim_classes)
            s_repeated = s.repeat(self.action_space.total_size(), 1, 1)
        else:
            s_repeated = s.repeat(self.action_space.total_size(), 1)
            
        h_repeated = h.repeat(self.action_space.total_size(), 1)
        
        all_actions = self.action_space.all_actions_as_tensor(device=s.device)
        s_next, dist_next, h_next = model.transition(s_repeated, all_actions, h_repeated)
        

        x_hat_next, x_hat_mask_next = model.decode(s_next, h_next)
        z_hat, dist_from_recon = model.encode(attach_goal_mask(x_hat_next, x_hat_mask_next), h_next)
        posterior = D.OneHotCategorical(logits=dist_next[0])
        posterior_given_o = D.OneHotCategorical(logits=dist_from_recon[0])
        x_hat_hat, x_hat_hat_mask = model.decode(z_hat, h_next)

        kl = D.kl_divergence(posterior, posterior_given_o)

        return kl, posterior, posterior_given_o, x_hat_next, x_hat_mask_next, x_hat_hat, x_hat_hat_mask