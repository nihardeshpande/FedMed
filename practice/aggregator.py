"""
FedMed Server Aggregator
Federated averaging with homomorphic encryption support.
"""

import torch
import tenseal as ts
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ClientUpdate:
    """Container for client model updates."""
    client_id: str
    weights: Optional[Dict[str, torch.Tensor]] = None
    encrypted_weights: Optional[bytes] = None
    num_samples: int = 0
    round_number: int = 0


class EncryptionStrategy(ABC):
    """Abstract base for encryption strategies."""
    
    @abstractmethod
    def encrypt(self, weights: Dict[str, torch.Tensor]) -> bytes:
        pass
    
    @abstractmethod
    def decrypt(self, encrypted: bytes) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def aggregate_encrypted(self, encrypted_list: List[bytes]) -> bytes:
        pass


class TenSEALEncryption(EncryptionStrategy):
    """Homomorphic encryption using TenSEAL."""
    
    def __init__(self, poly_modulus_degree: int = 8192):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
    
    def encrypt(self, weights: Dict[str, torch.Tensor]) -> bytes:
        flat = self._flatten(weights)
        encrypted = ts.ckks_vector(self.context, flat)
        return encrypted.serialize()
    
    def decrypt(self, encrypted: bytes) -> Dict[str, torch.Tensor]:
        vec = ts.ckks_vector_from(self.context, encrypted)
        return {"flattened": torch.tensor(vec.decrypt())}
    
    def aggregate_encrypted(self, encrypted_list: List[bytes]) -> bytes:
        result = ts.ckks_vector_from(self.context, encrypted_list[0])
        for enc in encrypted_list[1:]:
            result = result + ts.ckks_vector_from(self.context, enc)
        return result.serialize()
    
    def _flatten(self, weights: Dict[str, torch.Tensor]) -> List[float]:
        flat = []
        for t in weights.values():
            flat.extend(t.flatten().tolist())
        return flat


class FedAvgAggregator:
    """Federated Averaging with optional encryption."""
    
    def __init__(self, encryption=None, min_clients: int = 2):
        self.encryption = encryption
        self.min_clients = min_clients
        self.rounds = []
    
    def aggregate(self, updates: List[ClientUpdate], global_weights):
        if len(updates) < self.min_clients:
            raise ValueError(f"Need {self.min_clients} clients, got {len(updates)}")
        
        if self.encryption and all(u.encrypted_weights for u in updates):
            return self._encrypted_agg(updates)
        return self._plain_agg(updates, global_weights)
    
    def _plain_agg(self, updates, global_weights):
        total = sum(u.num_samples for u in updates)
        new_weights = {}
        
        for key in global_weights:
            new_weights[key] = sum(
                (u.num_samples / total) * u.weights[key] for u in updates
            )
        
        self.rounds.append({"clients": len(updates), "encrypted": False})
        return new_weights
    
    def _encrypted_agg(self, updates):
        enc_list = [u.encrypted_weights for u in updates]
        result = self.encryption.aggregate_encrypted(enc_list)
        weights = self.encryption.decrypt(result)
        
        # Average
        n = len(updates)
        weights["flattened"] = weights["flattened"] / n
        
        self.rounds.append({"clients": len(updates), "encrypted": True})
        return weights