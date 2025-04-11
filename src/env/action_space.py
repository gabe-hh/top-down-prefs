import numpy as np
import torch

class ActionSpace:
    def __init__(self, space_type, n=None, shape=None, low=None, high=None, num_dims=None, dim_size=None):
        """
        Initialize an ActionSpace.
        
        For discrete spaces, you can specify the space in two ways:
        
          1. **Composite Mode:**  
             - Provide either a single integer n (for a one-dimensional discrete space)  
               or a tuple 'shape'.  
             - For example, shape=(3,2) means the action is composite: one part has 3 options and the other has 2.
        
          2. **Independent Dimensions Mode:**  
             - Provide num_dims and dim_size.  
             - For example, num_dims=3 and dim_size=2 indicates 3 independent action dimensions, each with 2 options.
        
        For continuous spaces:
          - space_type: "continuous"
          - shape: shape of the continuous action vector.
          - low: lower bound(s) for the action values (scalar or array matching shape).
          - high: upper bound(s) for the action values.
        """
        self.space_type = space_type
        if self.space_type == "discrete":
            if num_dims is not None and dim_size is not None:
                # Use independent dimensions mode
                self.shape = (dim_size,) * num_dims
            elif n is not None:
                self.shape = (n,)
            elif shape is not None:
                self.shape = tuple(shape)
            else:
                raise ValueError("For discrete spaces, provide either n, shape, or (num_dims and dim_size).")
            # Total number of discrete actions (flat)
            self.n = int(np.prod(self.shape))
        elif self.space_type == "continuous":
            if shape is None:
                raise ValueError("For continuous spaces, shape must be provided.")
            self.shape = tuple(shape)
            self.low = low if low is not None else -1.0
            self.high = high if high is not None else 1.0
        else:
            raise ValueError("Unsupported space type. Use 'discrete' or 'continuous'.")

        self.cached_all_actions_as_tensor = None

    def total_size(self):
        """
        Returns the total number of discrete actions (flattened).
        Not applicable for continuous action spaces.
        """
        if self.space_type == "discrete":
            return self.n
        else:
            raise NotImplementedError("Continuous action spaces do not have a finite total size.")

    def multi_index_to_flat(self, multi_index):
        """
        Convert a multi-dimensional discrete action (tuple of indices) to a flat index.
        """
        if self.space_type != "discrete":
            raise ValueError("This method applies only to discrete action spaces.")
        multi_index = tuple(multi_index)
        if len(multi_index) != len(self.shape):
            raise ValueError("Dimension mismatch: expected {} indices.".format(len(self.shape)))
        return int(np.ravel_multi_index(multi_index, dims=self.shape))
    
    def flat_to_multi_index(self, flat_index):
        """
        Convert a flat index into its multi-dimensional index representation (tuple).
        """
        if self.space_type != "discrete":
            raise ValueError("This method applies only to discrete action spaces.")
        return tuple(np.unravel_index(flat_index, shape=self.shape))
    
    def index_to_one_hot(self, index, flatten=True):
        """
        Convert discrete action(s) into one-hot representation(s).
        
        If the action space is multi-dimensional and flatten is True,
        returns a flat one-hot vector of length total_size().
        If flatten is False, returns a list of one-hot vectors (one per dimension).
        
        The input index can be:
        - A single flat index (integer)
        - A single multi-index (tuple)
        - A list/array of flat indices
        - A list/array of multi-indices
        
        Returns:
        - For a single action: one-hot vector or list of vectors
        - For multiple actions: list of one-hot vectors or list of lists of vectors
        """
        if self.space_type != "discrete":
            raise ValueError("One-hot encoding applies only to discrete action spaces.")
        
        # Check if index is a list/array of actions
        if isinstance(index, (list, np.ndarray)) and len(index) > 0 and not (
                isinstance(index, (tuple, list, np.ndarray)) and 
                len(index) == len(self.shape) and all(isinstance(i, (int, np.integer)) for i in index)):
            # Process list of actions
            return [self.index_to_one_hot(idx, flatten) for idx in index]
        
        # Process single action
        total = self.total_size()
        one_hot_flat = np.zeros(total, dtype=np.float32)
        if isinstance(index, (list, tuple, np.ndarray)) and len(index) == len(self.shape):
            flat_index = self.multi_index_to_flat(index)
        else:
            flat_index = int(index)
        one_hot_flat[flat_index] = 1.0
        
        if flatten:
            return one_hot_flat
        else:
            # Return a list of one-hot vectors per dimension
            multi_idx = self.flat_to_multi_index(flat_index)
            one_hot_list = []
            for i, dim in enumerate(self.shape):
                one_hot_dim = np.zeros(dim, dtype=np.float32)
                one_hot_dim[multi_idx[i]] = 1.0
                one_hot_list.append(one_hot_dim)
            return one_hot_list

    def one_hot_to_index(self, one_hot, flattened=True):
        """
        Convert a one-hot representation back to the corresponding discrete action.
        
        If flattened is True, one_hot is assumed to be a 1D vector and the returned action is the flat index.
        If flattened is False, one_hot is assumed to be a list (or array) of one-hot vectors per dimension,
        and the returned action is a multi-index tuple.
        """
        if self.space_type != "discrete":
            raise ValueError("One-hot decoding applies only to discrete action spaces.")
        if flattened:
            return int(np.argmax(one_hot))
        else:
            return tuple(int(np.argmax(vec)) for vec in one_hot)

    def get_remaining_actions(self, taken):
        """
        Given a collection of taken actions, return a sorted list of the remaining flat indices.
        Each action in 'taken' can be provided either as a flat index or as a multi-index tuple.
        """
        if self.space_type != "discrete":
            raise ValueError("Remaining actions are defined only for discrete action spaces.")
        taken_flat = set()
        for act in taken:
            if isinstance(act, (list, tuple, np.ndarray)) and len(act) == len(self.shape):
                taken_flat.add(self.multi_index_to_flat(act))
            else:
                taken_flat.add(int(act))
        all_indices = set(range(self.total_size()))
        remaining = sorted(list(all_indices - taken_flat))
        return remaining
    
    def actions(self):
        """
        Return a list of all possible actions in the action space.
        """
        if self.space_type != "discrete":
            raise ValueError("All actions are defined only for discrete action spaces.")
        return list(range(self.total_size()))
    
    def tensor_to_index(self, tensor):
        """
        Convert a one-hot tensor to the corresponding discrete action index.
        """
        if self.space_type != "discrete":
            raise ValueError("Tensor conversion applies only to discrete action spaces.")
        if len(self.shape) > 1:
            # Multi-dimensional case
            one_hot = [vec.cpu().numpy() for vec in tensor]
            return self.one_hot_to_index(one_hot, flattened=False)
        else:
            # Single-dimensional case
            one_hot = tensor.cpu().numpy()
            return self.one_hot_to_index(one_hot, flattened=True)
    
    def index_to_tensor(self, index, device=None):
        """
        Convert discrete action index(es) to the corresponding one-hot tensor(s).
        
        Args:
            index: Either a single action index (int or tuple), a list/array of indices,
                   or a list of sequences (list of lists of indices)
            device: Optional torch device to place the tensors on
                
        Returns:
            A tensor representation of the action indices:
            - For a single index in a multi-dimensional space: tensor of shape [action_dimensions]
            - For a single index in a single-dimensional space: tensor of shape [action_size]
            - For multiple indices: tensor of shape [batch_size, action_size]
            - For a list of sequences: tensor of shape [sequence_length, num_sequences, action_size]
        """
        if self.space_type != "discrete":
            raise ValueError("Tensor conversion applies only to discrete action spaces.")
        
        # Check if we have a list of sequences
        is_sequence_batch = (isinstance(index, (list, tuple)) and len(index) > 0 and 
                             isinstance(index[0], (list, tuple, np.ndarray)))
        
        if is_sequence_batch:
            # Process list of sequences (list of lists)
            num_sequences = len(index)
            seq_length = max(len(seq) for seq in index)
            
            if len(self.shape) == 1:
                # Single-dimensional action space
                action_size = self.shape[0]
                result = torch.zeros((seq_length, num_sequences, action_size), dtype=torch.float32)
                
                for seq_idx, sequence in enumerate(index):
                    for step_idx, action_idx in enumerate(sequence):
                        if isinstance(action_idx, (int, np.integer)):
                            result[step_idx, seq_idx, action_idx] = 1.0
                        else:
                            # Handle tuple indices if provided
                            flat_idx = self.multi_index_to_flat(action_idx) if isinstance(action_idx, (tuple, list)) else action_idx
                            result[step_idx, seq_idx, flat_idx] = 1.0
            else:
                # Multi-dimensional action space - convert to flattened representation
                action_size = self.total_size()
                result = torch.zeros((seq_length, num_sequences, action_size), dtype=torch.float32)
                
                for seq_idx, sequence in enumerate(index):
                    for step_idx, action_idx in enumerate(sequence):
                        if isinstance(action_idx, (tuple, list)) and len(action_idx) == len(self.shape):
                            flat_idx = self.multi_index_to_flat(action_idx)
                        else:
                            flat_idx = action_idx
                        result[step_idx, seq_idx, flat_idx] = 1.0
        else:
            # Handle single index or batch of indices (original behavior but with tensor output)
            is_batch = isinstance(index, (list, tuple, np.ndarray)) and not (
                isinstance(index, (list, tuple, np.ndarray)) and 
                len(index) > 0 and isinstance(index[0], (list, tuple, np.ndarray)) and 
                len(index[0]) == len(self.shape)
            )
            
            if is_batch:
                # Batch of indices
                batch_size = len(index)
                if len(self.shape) == 1:
                    # Single-dimensional case
                    result = torch.zeros((batch_size, self.shape[0]), dtype=torch.float32)
                    for i, idx in enumerate(index):
                        result[i, idx] = 1.0
                else:
                    # Multi-dimensional case
                    result = torch.zeros((batch_size, self.total_size()), dtype=torch.float32)
                    for i, idx in enumerate(index):
                        if isinstance(idx, (tuple, list)) and len(idx) == len(self.shape):
                            flat_idx = self.multi_index_to_flat(idx)
                        else:
                            flat_idx = idx
                        result[i, flat_idx] = 1.0
            else:
                # Single index
                if len(self.shape) == 1:
                    # Single-dimensional case
                    one_hot = self.index_to_one_hot(index, flatten=True)
                    result = torch.tensor(one_hot, dtype=torch.float32)
                else:
                    # Multi-dimensional case - return flattened representation
                    one_hot = self.index_to_one_hot(index, flatten=True)
                    result = torch.tensor(one_hot, dtype=torch.float32)
        
        # Move to specified device if needed
        if device is not None:
            result = result.to(device)
            
        return result
    
    def all_actions_as_tensor(self, device=None):
        """
        Return all possible actions as one-hot tensors.
        """
        if self.space_type != "discrete":
            raise ValueError("One-hot tensors are defined only for discrete action spaces.")
            
        # Use cached result if available and device matches
        if self.cached_all_actions_as_tensor is not None:
            # If device is specified and we need to move the tensor
            if device and hasattr(self.cached_all_actions_as_tensor, 'to') and str(self.cached_all_actions_as_tensor.device) != str(device):
                return self.cached_all_actions_as_tensor.to(device)
            return self.cached_all_actions_as_tensor

        total_actions = self.total_size()
        all_actions = []

        for i in range(total_actions):
            # Get one-hot representation for this action
            if len(self.shape) > 1:
                # Multi-dimensional case
                one_hot = self.index_to_one_hot(i, flatten=False)
                one_hot_tensors = [torch.tensor(oh, dtype=torch.float32) for oh in one_hot]
                all_actions.append(one_hot_tensors)
            else:
                # Single-dimensional case
                one_hot = self.index_to_one_hot(i, flatten=True)
                all_actions.append(torch.tensor(one_hot, dtype=torch.float32))

        # Create final tensor structure
        if len(self.shape) > 1:
            # For multi-dimensional case, create a list of tensors per dimension
            result = [torch.stack([act[dim] for act in all_actions]) for dim in range(len(self.shape))]
        else:
            # For single-dimensional case, stack into a single tensor
            result = torch.stack(all_actions)

        # Cache the result
        self.cached_all_actions_as_tensor = result

        # Move to specified device if needed
        if device and hasattr(result, 'to'):
            result = result.to(device)
        elif device and isinstance(result, list):
            result = [r.to(device) for r in result]

        return result

    def sample(self, probs=None):
        """
        Randomly sample an action from the action space.
        
        For discrete spaces:
          - If 'probs' is provided, it should be a flat array-like of probabilities for each action.
          - Returns both a flat index and its corresponding multi-index tuple.
        
        For continuous spaces:
          - Returns a numpy array of the appropriate shape.
        """
        if self.space_type == "discrete":
            total = self.total_size()
            if probs is not None:
                probs = np.array(probs, dtype=np.float64)
                if probs.shape[0] != total:
                    raise ValueError("Length of probs must match the total number of discrete actions.")
                # Ensure the probabilities sum to 1.
                probs = probs / probs.sum()
                flat_index = np.random.choice(np.arange(total), p=probs)
            else:
                flat_index = np.random.randint(0, total)
            multi_index = self.flat_to_multi_index(flat_index)
            return flat_index, multi_index
        elif self.space_type == "continuous":
            if np.isscalar(self.low) and np.isscalar(self.high):
                return np.random.uniform(self.low, self.high, size=self.shape).astype(np.float32)
            else:
                low_arr = np.array(self.low)
                high_arr = np.array(self.high)
                return np.random.uniform(low_arr, high_arr, size=self.shape).astype(np.float32)
        else:
            raise ValueError("Unsupported action space type.")

# --- Example Usage ---

if __name__ == '__main__':
    # Example 1: Composite discrete action space with shape=(3,2)
    composite_space = ActionSpace(space_type="discrete", shape=(3,2))
    print("Composite Discrete total size:", composite_space.total_size())
    # Here, the space is interpreted as a composite: one part of size 3 and one part of size 2.
    index = 2
    flat_one_hot = composite_space.index_to_one_hot(index, flatten=True)
    print("Flat one-hot for flat index {}: {}".format(index, flat_one_hot))
    multi_index = composite_space.flat_to_multi_index(index)
    print("Multi-index for flat index {}: {}".format(index, multi_index))
    
    # Example 2: Independent dimensions discrete action space: 3 dimensions of size 2 each.
    independent_space = ActionSpace(space_type="discrete", num_dims=4, dim_size=4)
    print("\nIndependent Discrete total size:", independent_space.total_size())
    # Now, the space is interpreted as 3 independent dimensions, each with 2 possible actions.
    index = 5  # For example, flat index 5 out of 2*2*2 = 8 actions.
    flat_one_hot = independent_space.index_to_one_hot(index, flatten=True)
    print("Flat one-hot for flat index {}: {}".format(index, flat_one_hot))
    multi_index = independent_space.flat_to_multi_index(index)
    print("Multi-index for flat index {}: {}".format(index, multi_index))
    # Also show per-dimension one-hot if desired:
    per_dim_one_hot = independent_space.index_to_one_hot(multi_index, flatten=False)
    print("Per-dimension one-hot for multi-index {}: {}".format(multi_index, per_dim_one_hot))
    
    # Example 3: Sampling
    flat_idx, multi_idx = independent_space.sample()
    print("\nSampled independent discrete action: flat index {}, multi-index {}".format(flat_idx, multi_idx))
    
    # Example 4: Continuous action space (3-dimensional, e.g., joint angles)
    continuous_space = ActionSpace(space_type="continuous", shape=(3,), low=-2.0, high=2.0)
    print("\nSampled continuous action:", continuous_space.sample())
