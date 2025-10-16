"""
Neural Architecture Search (NAS) Module

Automated neural architecture design for optimal model discovery.
Supports multiple search strategies and performance prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import random
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy types"""
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DIFFERENTIABLE = "differentiable"


class OperationType(Enum):
    """Neural network operation types"""
    CONV_3X3 = "conv_3x3"
    CONV_5X5 = "conv_5x5"
    CONV_7X7 = "conv_7x7"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    SKIP_CONNECT = "skip_connect"
    DENSE = "dense"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RELU = "relu"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"


@dataclass
class ArchitectureConfig:
    """Configuration for a neural architecture"""
    layers: List[Dict[str, Any]]
    input_size: int
    output_size: int
    num_layers: int
    hidden_sizes: List[int]
    operations: List[str]
    connections: List[Tuple[int, int]]  # (from_layer, to_layer)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchitectureConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def __hash__(self):
        """Make hashable for caching"""
        return hash(json.dumps(self.to_dict(), sort_keys=True))


@dataclass
class ArchitecturePerformance:
    """Performance metrics for an architecture"""
    accuracy: float
    latency: float  # milliseconds
    params: int  # number of parameters
    flops: int  # floating point operations
    memory: float  # MB
    training_time: float  # seconds
    
    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted score"""
        if weights is None:
            weights = {
                'accuracy': 0.5,
                'latency': 0.2,
                'params': 0.15,
                'memory': 0.15
            }
        
        # Normalize metrics (higher is better)
        norm_accuracy = self.accuracy
        norm_latency = 1.0 / (self.latency + 1e-6)
        norm_params = 1.0 / (self.params + 1e-6)
        norm_memory = 1.0 / (self.memory + 1e-6)
        
        score = (
            weights['accuracy'] * norm_accuracy +
            weights['latency'] * norm_latency +
            weights['params'] * norm_params +
            weights['memory'] * norm_memory
        )
        return score


class SearchSpace:
    """Define the search space for NAS"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        max_layers: int = 10,
        min_layers: int = 2,
        hidden_size_range: Tuple[int, int] = (32, 512)
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.hidden_size_range = hidden_size_range
        
        # Available operations
        self.operations = [op.value for op in OperationType]
        
        logger.info(f"Search space: {min_layers}-{max_layers} layers, "
                   f"hidden sizes {hidden_size_range}, "
                   f"{len(self.operations)} operations")
    
    def sample_architecture(self) -> ArchitectureConfig:
        """Sample a random architecture from search space"""
        num_layers = random.randint(self.min_layers, self.max_layers)
        
        # Sample hidden sizes
        hidden_sizes = [
            random.randint(*self.hidden_size_range)
            for _ in range(num_layers)
        ]
        
        # Sample operations for each layer
        operations = [
            random.choice(self.operations)
            for _ in range(num_layers)
        ]
        
        # Create layer configurations
        layers = []
        for i in range(num_layers):
            layer = {
                'id': i,
                'operation': operations[i],
                'hidden_size': hidden_sizes[i],
                'activation': random.choice(['relu', 'tanh', 'sigmoid'])
            }
            layers.append(layer)
        
        # Sample skip connections (optional)
        connections = []
        for i in range(num_layers):
            # 30% chance of skip connection to previous layers
            if i > 0 and random.random() < 0.3:
                from_layer = random.randint(0, i - 1)
                connections.append((from_layer, i))
        
        return ArchitectureConfig(
            layers=layers,
            input_size=self.input_size,
            output_size=self.output_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            operations=operations,
            connections=connections
        )
    
    def mutate_architecture(
        self,
        config: ArchitectureConfig,
        mutation_rate: float = 0.1
    ) -> ArchitectureConfig:
        """Mutate an architecture for evolutionary search"""
        new_config = ArchitectureConfig(**config.to_dict())
        
        # Mutate operations
        for i, layer in enumerate(new_config.layers):
            if random.random() < mutation_rate:
                layer['operation'] = random.choice(self.operations)
                layer['activation'] = random.choice(['relu', 'tanh', 'sigmoid'])
        
        # Mutate hidden sizes
        for i in range(len(new_config.hidden_sizes)):
            if random.random() < mutation_rate:
                new_config.hidden_sizes[i] = random.randint(*self.hidden_size_range)
                new_config.layers[i]['hidden_size'] = new_config.hidden_sizes[i]
        
        # Mutate connections
        if random.random() < mutation_rate:
            if new_config.connections and random.random() < 0.5:
                # Remove a connection
                new_config.connections.pop(random.randint(0, len(new_config.connections) - 1))
            else:
                # Add a connection
                i = random.randint(1, new_config.num_layers - 1)
                from_layer = random.randint(0, i - 1)
                new_config.connections.append((from_layer, i))
        
        return new_config
    
    def crossover_architectures(
        self,
        config1: ArchitectureConfig,
        config2: ArchitectureConfig
    ) -> Tuple[ArchitectureConfig, ArchitectureConfig]:
        """Crossover two architectures for evolutionary search"""
        # Simple single-point crossover
        min_layers = min(config1.num_layers, config2.num_layers)
        crossover_point = random.randint(1, min_layers - 1)
        
        # Create offspring
        offspring1_layers = (
            config1.layers[:crossover_point] +
            config2.layers[crossover_point:config2.num_layers]
        )
        offspring2_layers = (
            config2.layers[:crossover_point] +
            config1.layers[crossover_point:config1.num_layers]
        )
        
        offspring1 = ArchitectureConfig(
            layers=offspring1_layers,
            input_size=self.input_size,
            output_size=self.output_size,
            num_layers=len(offspring1_layers),
            hidden_sizes=[l['hidden_size'] for l in offspring1_layers],
            operations=[l['operation'] for l in offspring1_layers],
            connections=config1.connections
        )
        
        offspring2 = ArchitectureConfig(
            layers=offspring2_layers,
            input_size=self.input_size,
            output_size=self.output_size,
            num_layers=len(offspring2_layers),
            hidden_sizes=[l['hidden_size'] for l in offspring2_layers],
            operations=[l['operation'] for l in offspring2_layers],
            connections=config2.connections
        )
        
        return offspring1, offspring2


class PerformancePredictor:
    """Predict architecture performance without full training"""
    
    def __init__(self):
        self.history: Dict[int, ArchitecturePerformance] = {}
        self.predictor_model = None
        
    def predict(self, config: ArchitectureConfig) -> ArchitecturePerformance:
        """Predict performance of an architecture"""
        # Check cache
        config_hash = hash(config)
        if config_hash in self.history:
            return self.history[config_hash]
        
        # Simple heuristic-based prediction
        # In production, this would be a trained surrogate model
        
        # Estimate parameters
        params = self._estimate_params(config)
        
        # Estimate FLOPs
        flops = self._estimate_flops(config)
        
        # Estimate memory
        memory = params * 4 / (1024 * 1024)  # 4 bytes per float32, convert to MB
        
        # Estimate latency (ms)
        latency = (flops / 1e9) * 10  # rough estimate
        
        # Estimate accuracy (heuristic)
        # More layers and parameters generally better, but with diminishing returns
        accuracy_score = min(0.95, 0.7 + 0.05 * np.log(params / 1000))
        
        # Add some noise for realism
        accuracy_score += np.random.normal(0, 0.02)
        accuracy_score = np.clip(accuracy_score, 0.0, 1.0)
        
        # Estimate training time
        training_time = (params / 1e6) * 100  # rough estimate in seconds
        
        performance = ArchitecturePerformance(
            accuracy=accuracy_score,
            latency=latency,
            params=params,
            flops=flops,
            memory=memory,
            training_time=training_time
        )
        
        # Cache result
        self.history[config_hash] = performance
        
        return performance
    
    def _estimate_params(self, config: ArchitectureConfig) -> int:
        """Estimate number of parameters"""
        params = 0
        prev_size = config.input_size
        
        for layer in config.layers:
            hidden_size = layer['hidden_size']
            op = layer['operation']
            
            if 'conv' in op:
                # Conv layers: kernel_size^2 * in_channels * out_channels
                kernel_size = int(op.split('_')[1].replace('x', ''))
                params += kernel_size * kernel_size * prev_size * hidden_size
            elif op == 'dense':
                # Dense layers: in_features * out_features
                params += prev_size * hidden_size
            elif op in ['lstm', 'gru']:
                # RNN layers: roughly 4 * hidden_size^2 for LSTM
                params += 4 * hidden_size * hidden_size
            
            prev_size = hidden_size
        
        # Output layer
        params += prev_size * config.output_size
        
        return params
    
    def _estimate_flops(self, config: ArchitectureConfig) -> int:
        """Estimate floating point operations"""
        flops = 0
        prev_size = config.input_size
        
        for layer in config.layers:
            hidden_size = layer['hidden_size']
            op = layer['operation']
            
            if 'conv' in op:
                kernel_size = int(op.split('_')[1].replace('x', ''))
                flops += kernel_size * kernel_size * prev_size * hidden_size
            elif op == 'dense':
                flops += 2 * prev_size * hidden_size
            elif op in ['lstm', 'gru']:
                flops += 8 * hidden_size * hidden_size
            
            prev_size = hidden_size
        
        return flops
    
    def update_with_actual(
        self,
        config: ArchitectureConfig,
        performance: ArchitecturePerformance
    ):
        """Update predictor with actual measured performance"""
        config_hash = hash(config)
        self.history[config_hash] = performance
        logger.info(f"Updated predictor with actual performance: "
                   f"accuracy={performance.accuracy:.4f}")


class NeuralArchitectureSearch:
    """
    Neural Architecture Search engine
    
    Supports multiple search strategies:
    - Random search
    - Evolutionary algorithm
    - Reinforcement learning (simplified)
    - Differentiable NAS (DARTS-like)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.search_space = search_space
        self.strategy = strategy
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.predictor = PerformancePredictor()
        self.best_architectures: List[Tuple[ArchitectureConfig, ArchitecturePerformance]] = []
        self.search_history: List[Dict] = []
        
        logger.info(f"NAS initialized with {strategy.value} strategy")
    
    def search(
        self,
        max_evaluations: int = 100,
        performance_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[ArchitectureConfig, ArchitecturePerformance]:
        """
        Run architecture search
        
        Returns:
            Best architecture and its performance
        """
        logger.info(f"Starting NAS with {self.strategy.value} strategy, "
                   f"max_evaluations={max_evaluations}")
        
        start_time = time.time()
        
        if self.strategy == SearchStrategy.RANDOM:
            best = self._random_search(max_evaluations, performance_weights)
        elif self.strategy == SearchStrategy.EVOLUTIONARY:
            best = self._evolutionary_search(max_evaluations, performance_weights)
        elif self.strategy == SearchStrategy.REINFORCEMENT_LEARNING:
            best = self._rl_search(max_evaluations, performance_weights)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        search_time = time.time() - start_time
        logger.info(f"NAS completed in {search_time:.2f}s")
        logger.info(f"Best architecture: accuracy={best[1].accuracy:.4f}, "
                   f"params={best[1].params:,}, latency={best[1].latency:.2f}ms")
        
        return best
    
    def _random_search(
        self,
        max_evaluations: int,
        performance_weights: Optional[Dict[str, float]]
    ) -> Tuple[ArchitectureConfig, ArchitecturePerformance]:
        """Random search strategy"""
        best_config = None
        best_performance = None
        best_score = -float('inf')
        
        for i in range(max_evaluations):
            # Sample random architecture
            config = self.search_space.sample_architecture()
            
            # Evaluate
            performance = self.predictor.predict(config)
            score = performance.score(performance_weights)
            
            # Track best
            if score > best_score:
                best_score = score
                best_config = config
                best_performance = performance
                logger.info(f"Iteration {i+1}/{max_evaluations}: "
                           f"New best score={score:.4f}")
            
            # Record history
            self.search_history.append({
                'iteration': i,
                'config': config.to_dict(),
                'performance': asdict(performance),
                'score': score
            })
        
        self.best_architectures.append((best_config, best_performance))
        return best_config, best_performance
    
    def _evolutionary_search(
        self,
        max_evaluations: int,
        performance_weights: Optional[Dict[str, float]]
    ) -> Tuple[ArchitectureConfig, ArchitecturePerformance]:
        """Evolutionary search strategy"""
        # Initialize population
        population = [
            self.search_space.sample_architecture()
            for _ in range(self.population_size)
        ]
        
        # Evaluate initial population
        fitness_scores = []
        for config in population:
            performance = self.predictor.predict(config)
            score = performance.score(performance_weights)
            fitness_scores.append((config, performance, score))
        
        best_config = None
        best_performance = None
        best_score = -float('inf')
        
        evaluations = self.population_size
        generation = 0
        
        while evaluations < max_evaluations:
            generation += 1
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Track best
            if fitness_scores[0][2] > best_score:
                best_score = fitness_scores[0][2]
                best_config = fitness_scores[0][0]
                best_performance = fitness_scores[0][1]
                logger.info(f"Generation {generation}: "
                           f"Best score={best_score:.4f}")
            
            # Selection: keep top 50%
            survivors = fitness_scores[:self.population_size // 2]
            
            # Create new population
            new_population = [config for config, _, _ in survivors]
            
            # Crossover
            while len(new_population) < self.population_size and evaluations < max_evaluations:
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(survivors, 2)
                    offspring1, offspring2 = self.search_space.crossover_architectures(
                        parent1[0], parent2[0]
                    )
                    new_population.extend([offspring1, offspring2])
                    evaluations += 2
            
            # Mutation
            mutated_population = []
            for config in new_population[:self.population_size]:
                if random.random() < self.mutation_rate:
                    config = self.search_space.mutate_architecture(
                        config, self.mutation_rate
                    )
                mutated_population.append(config)
                evaluations += 1
            
            # Evaluate new population
            fitness_scores = []
            for config in mutated_population[:self.population_size]:
                performance = self.predictor.predict(config)
                score = performance.score(performance_weights)
                fitness_scores.append((config, performance, score))
        
        self.best_architectures.append((best_config, best_performance))
        return best_config, best_performance
    
    def _rl_search(
        self,
        max_evaluations: int,
        performance_weights: Optional[Dict[str, float]]
    ) -> Tuple[ArchitectureConfig, ArchitecturePerformance]:
        """Reinforcement learning-based search (simplified)"""
        # Simplified RL: treat as multi-armed bandit problem
        # Track rewards for each operation type
        operation_rewards = defaultdict(list)
        
        best_config = None
        best_performance = None
        best_score = -float('inf')
        
        # Exploration phase
        exploration_budget = max_evaluations // 3
        for i in range(exploration_budget):
            config = self.search_space.sample_architecture()
            performance = self.predictor.predict(config)
            score = performance.score(performance_weights)
            
            # Update rewards for operations used
            for layer in config.layers:
                operation_rewards[layer['operation']].append(score)
            
            if score > best_score:
                best_score = score
                best_config = config
                best_performance = performance
        
        # Exploitation phase: bias towards high-reward operations
        for i in range(exploration_budget, max_evaluations):
            # Sample with bias towards good operations
            config = self._sample_with_bias(operation_rewards)
            performance = self.predictor.predict(config)
            score = performance.score(performance_weights)
            
            # Update rewards
            for layer in config.layers:
                operation_rewards[layer['operation']].append(score)
            
            if score > best_score:
                best_score = score
                best_config = config
                best_performance = performance
                logger.info(f"Iteration {i+1}/{max_evaluations}: "
                           f"New best score={score:.4f}")
        
        self.best_architectures.append((best_config, best_performance))
        return best_config, best_performance
    
    def _sample_with_bias(
        self,
        operation_rewards: Dict[str, List[float]]
    ) -> ArchitectureConfig:
        """Sample architecture biased towards high-reward operations"""
        # Calculate average reward for each operation
        avg_rewards = {
            op: np.mean(rewards) if rewards else 0.0
            for op, rewards in operation_rewards.items()
        }
        
        # Softmax to get probabilities
        ops = list(avg_rewards.keys())
        rewards = np.array([avg_rewards[op] for op in ops])
        exp_rewards = np.exp(rewards - np.max(rewards))
        probs = exp_rewards / exp_rewards.sum()
        
        # Sample architecture with biased operations
        num_layers = random.randint(
            self.search_space.min_layers,
            self.search_space.max_layers
        )
        
        operations = np.random.choice(ops, size=num_layers, p=probs).tolist()
        hidden_sizes = [
            random.randint(*self.search_space.hidden_size_range)
            for _ in range(num_layers)
        ]
        
        layers = []
        for i in range(num_layers):
            layer = {
                'id': i,
                'operation': operations[i],
                'hidden_size': hidden_sizes[i],
                'activation': 'relu'
            }
            layers.append(layer)
        
        return ArchitectureConfig(
            layers=layers,
            input_size=self.search_space.input_size,
            output_size=self.search_space.output_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            operations=operations,
            connections=[]
        )
    
    def get_top_k_architectures(
        self,
        k: int = 5
    ) -> List[Tuple[ArchitectureConfig, ArchitecturePerformance]]:
        """Get top k best architectures found"""
        sorted_archs = sorted(
            self.best_architectures,
            key=lambda x: x[1].score(),
            reverse=True
        )
        return sorted_archs[:k]
    
    def export_search_results(self, filepath: str):
        """Export search results to file"""
        results = {
            'strategy': self.strategy.value,
            'best_architectures': [
                {
                    'config': config.to_dict(),
                    'performance': asdict(performance)
                }
                for config, performance in self.best_architectures
            ],
            'search_history': self.search_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Search results exported to {filepath}")
