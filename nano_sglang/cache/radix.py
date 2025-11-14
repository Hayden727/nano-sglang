"""Simplified RadixAttention prefix cache implementation."""

import time
from collections import defaultdict
from typing import List, Optional, Tuple, Dict
import torch


class RadixKey:
    """Key for radix tree, containing token IDs and optional extra key."""
    
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        self.token_ids = token_ids
        self.extra_key = extra_key
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)
    
    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        suffix = '...' if len(self.token_ids) > 10 else ''
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{suffix})"


class TreeNode:
    """Node in the radix tree."""
    
    counter = 0
    
    def __init__(self, node_id: Optional[int] = None):
        self.children: Dict[Tuple, 'TreeNode'] = {}
        self.parent: Optional['TreeNode'] = None
        self.key: Optional[RadixKey] = None
        # Value stores KV cache indices for this prefix
        self.value: Optional[List[int]] = None
        self.lock_ref = 0  # Reference counter to prevent eviction
        self.last_access_time = time.monotonic()
        self.hit_count = 0
        
        self.id = TreeNode.counter if node_id is None else node_id
        TreeNode.counter += 1
    
    @property
    def evicted(self) -> bool:
        return self.value is None
    
    def __lt__(self, other: 'TreeNode') -> bool:
        return self.last_access_time < other.last_access_time


class RadixCache:
    """
    Simplified RadixAttention prefix cache.
    
    This cache stores KV cache indices for token prefixes, allowing
    multiple requests with the same prefix to share cached KV values.
    """
    
    def __init__(self, max_size: int = 10000, eviction_policy: str = "lru"):
        """
        Initialize the radix cache.
        
        Args:
            max_size: Maximum number of cached prefixes
            eviction_policy: Eviction policy ("lru" or "fifo")
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.lock_ref = 1  # Root is always locked
        
        # Track all nodes for eviction
        self.all_nodes: List[TreeNode] = [self.root_node]
        self.evictable_nodes: List[TreeNode] = []
    
    def match_prefix(self, key: RadixKey) -> Tuple[List[int], TreeNode]:
        """
        Find the longest cached prefix of the given key.
        
        Args:
            key: RadixKey to match
            
        Returns:
            Tuple of (matched_indices, last_node)
            - matched_indices: List of KV cache indices for the matched prefix
            - last_node: The TreeNode where matching stopped
        """
        node = self.root_node
        matched_indices: List[int] = []
        
        i = 0
        while i < len(key.token_ids):
            # Get child key (single token for simplicity)
            child_key = key.token_ids[i]
            
            # Check if child exists
            if child_key not in node.children:
                break
            
            child = node.children[child_key]
            
            # Match as much as possible from child's key
            match_len = self._match_length(child.key, key[i:])
            
            if match_len == 0:
                break
            
            # If we matched part of child's key, we need to split
            if match_len < len(child.key):
                child = self._split_node(child, match_len)
            
            # Add child's value to matched indices
            if child.value is not None:
                matched_indices.extend(child.value)
            
            # Update access time
            child.last_access_time = time.monotonic()
            child.hit_count += 1
            
            node = child
            i += match_len
        
        return matched_indices, node
    
    def insert(
        self,
        key: RadixKey,
        kv_indices: List[int],
        last_node: Optional[TreeNode] = None
    ) -> TreeNode:
        """
        Insert a new prefix into the cache.
        
        Args:
            key: RadixKey to insert
            kv_indices: KV cache indices for this prefix
            last_node: Node to start insertion from (from match_prefix)
            
        Returns:
            The inserted node
        """
        if last_node is None:
            last_node = self.root_node
        
        # Find where to insert
        matched_indices, insert_node = self.match_prefix(key)
        
        # Calculate remaining tokens to insert
        remaining_tokens = key.token_ids[len(matched_indices):]
        
        if not remaining_tokens:
            # Prefix already exists, update value
            if insert_node.value is None:
                insert_node.value = kv_indices
                if insert_node not in self.evictable_nodes:
                    self.evictable_nodes.append(insert_node)
            return insert_node
        
        # Insert remaining tokens
        current_node = insert_node
        for i, token_id in enumerate(remaining_tokens):
            child_key = token_id
            
            if child_key not in current_node.children:
                # Create new node
                new_node = TreeNode()
                new_node.parent = current_node
                new_node.key = RadixKey(
                    token_ids=[token_id],
                    extra_key=key.extra_key
                )
                current_node.children[child_key] = new_node
                self.all_nodes.append(new_node)
                current_node = new_node
            else:
                current_node = current_node.children[child_key]
        
        # Set value for the final node
        current_node.value = kv_indices
        if current_node not in self.evictable_nodes:
            self.evictable_nodes.append(current_node)
        
        # Evict if necessary
        self._maybe_evict()
        
        return current_node
    
    def _split_node(self, node: TreeNode, split_pos: int) -> TreeNode:
        """
        Split a node at the given position.
        
        Args:
            node: Node to split
            split_pos: Position to split at
            
        Returns:
            The new node containing the matched portion
        """
        # Create new node for the matched portion
        new_node = TreeNode()
        new_node.parent = node.parent
        new_node.key = node.key[:split_pos]
        new_node.value = node.value[:split_pos] if node.value else None
        
        # Update original node
        node.key = node.key[split_pos:]
        if node.value:
            node.value = node.value[split_pos:]
        
        # Update parent's children
        if node.parent:
            # Find the key that points to node
            for k, v in node.parent.children.items():
                if v == node:
                    node.parent.children[k] = new_node
                    break
        
        # Update node's parent
        node.parent = new_node
        
        # Move children
        new_node.children = {node.key.token_ids[0]: node} if len(node.key) > 0 else {}
        
        self.all_nodes.append(new_node)
        return new_node
    
    def _match_length(self, key1: RadixKey, key2: RadixKey) -> int:
        """Calculate the length of matching prefix between two keys."""
        if key1.extra_key != key2.extra_key:
            return 0
        
        min_len = min(len(key1), len(key2))
        for i in range(min_len):
            if key1.token_ids[i] != key2.token_ids[i]:
                return i
        return min_len
    
    def _maybe_evict(self):
        """Evict nodes if cache is too large."""
        if len(self.evictable_nodes) <= self.max_size:
            return
        
        # Sort by eviction policy
        if self.eviction_policy == "lru":
            self.evictable_nodes.sort(key=lambda n: n.last_access_time)
        elif self.eviction_policy == "fifo":
            self.evictable_nodes.sort(key=lambda n: n.id)
        
        # Evict nodes that are not locked
        to_evict = []
        for node in self.evictable_nodes:
            if node.lock_ref == 0 and node != self.root_node:
                to_evict.append(node)
                if len(self.evictable_nodes) - len(to_evict) <= self.max_size:
                    break
        
        # Remove evicted nodes
        for node in to_evict:
            node.value = None
            self.evictable_nodes.remove(node)
    
    def reset(self):
        """Reset the cache."""
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.all_nodes = [self.root_node]
        self.evictable_nodes = []

