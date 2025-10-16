"""
Tests for Neural Architecture Search module
"""

import pytest
import numpy as np
import json
import tempfile
import os
from automl.nas import (
    NeuralArchitectureSearch,
    SearchSpace,
    PerformancePredictor,
    ArchitectureConfig,
    ArchitecturePerformance,
    SearchStrategy,
    OperationType
)


class TestArchitectureConfig:
    """Test ArchitectureConfig"""
    
    def test_create_config(self):
        """Test creating architecture config"""
        config = ArchitectureConfig(
            layers=[
                {'id': 0, 'operation': 'conv_3x3', 'hidden_size': 64, 'activation': 'relu'},
                {'id': 1, 'operation': 'dense', 'hidden_size': 128, 'activation': 'relu'}
            ],
            input_size=100,
            output_size=10,
            num_layers=2,
            hidden_sizes=[64, 128],
            operations=['conv_3x3', 'dense'],
            connections=[]
        )
        
        assert config.num_layers == 2
        assert len(config.layers) == 2
        assert config.input_size == 100
        assert config.output_size == 10
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = ArchitectureConfig(
            layers=[{'id': 0, 'operation': 'dense', 'hidden_size': 64, 'activation': 'relu'}],
            input_size=100,
            output_size=10,
            num_layers=1,
            hidden_sizes=[64],
            operations=['dense'],
            connections=[]
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'layers' in config_dict
        assert 'input_size' in config_dict
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'layers': [{'id': 0, 'operation': 'dense', 'hidden_size': 64, 'activation': 'relu'}],
            'input_size': 100,
            'output_size': 10,
            'num_layers': 1,
            'hidden_sizes': [64],
            'operations': ['dense'],
            'connections': []
        }
        
        config = ArchitectureConfig.from_dict(config_dict)
        assert config.num_layers == 1
        assert config.input_size == 100
    
    def test_config_hashable(self):
        """Test config is hashable"""
        config = ArchitectureConfig(
            layers=[{'id': 0, 'operation': 'dense', 'hidden_size': 64, 'activation': 'relu'}],
            input_size=100,
            output_size=10,
            num_layers=1,
            hidden_sizes=[64],
            operations=['dense'],
            connections=[]
        )
        
        hash_value = hash(config)
        assert isinstance(hash_value, int)


class TestArchitecturePerformance:
    """Test ArchitecturePerformance"""
    
    def test_create_performance(self):
        """Test creating performance metrics"""
        perf = ArchitecturePerformance(
            accuracy=0.95,
            latency=5.0,
            params=1000000,
            flops=5000000,
            memory=10.0,
            training_time=100.0
        )
        
        assert perf.accuracy == 0.95
        assert perf.latency == 5.0
        assert perf.params == 1000000
    
    def test_performance_score_default(self):
        """Test default performance scoring"""
        perf = ArchitecturePerformance(
            accuracy=0.95,
            latency=5.0,
            params=1000000,
            flops=5000000,
            memory=10.0,
            training_time=100.0
        )
        
        score = perf.score()
        assert isinstance(score, float)
        assert score > 0
    
    def test_performance_score_custom_weights(self):
        """Test custom weighted scoring"""
        perf = ArchitecturePerformance(
            accuracy=0.95,
            latency=5.0,
            params=1000000,
            flops=5000000,
            memory=10.0,
            training_time=100.0
        )
        
        weights = {
            'accuracy': 0.8,
            'latency': 0.1,
            'params': 0.05,
            'memory': 0.05
        }
        
        score = perf.score(weights)
        assert isinstance(score, float)
        assert score > 0


class TestSearchSpace:
    """Test SearchSpace"""
    
    def test_create_search_space(self):
        """Test creating search space"""
        space = SearchSpace(
            input_size=100,
            output_size=10,
            max_layers=10,
            min_layers=2,
            hidden_size_range=(32, 512)
        )
        
        assert space.input_size == 100
        assert space.output_size == 10
        assert space.max_layers == 10
        assert space.min_layers == 2
    
    def test_sample_architecture(self):
        """Test sampling random architecture"""
        space = SearchSpace(input_size=100, output_size=10)
        
        config = space.sample_architecture()
        
        assert isinstance(config, ArchitectureConfig)
        assert config.input_size == 100
        assert config.output_size == 10
        assert space.min_layers <= config.num_layers <= space.max_layers
    
    def test_sample_multiple_architectures(self):
        """Test sampling multiple architectures"""
        space = SearchSpace(input_size=100, output_size=10)
        
        configs = [space.sample_architecture() for _ in range(10)]
        
        assert len(configs) == 10
        # Check diversity
        unique_configs = set(hash(c) for c in configs)
        assert len(unique_configs) > 1  # Should have variety
    
    def test_mutate_architecture(self):
        """Test architecture mutation"""
        space = SearchSpace(input_size=100, output_size=10)
        
        original = space.sample_architecture()
        mutated = space.mutate_architecture(original, mutation_rate=0.5)
        
        assert isinstance(mutated, ArchitectureConfig)
        assert mutated.input_size == original.input_size
        assert mutated.output_size == original.output_size
    
    def test_crossover_architectures(self):
        """Test architecture crossover"""
        space = SearchSpace(input_size=100, output_size=10)
        
        config1 = space.sample_architecture()
        config2 = space.sample_architecture()
        
        offspring1, offspring2 = space.crossover_architectures(config1, config2)
        
        assert isinstance(offspring1, ArchitectureConfig)
        assert isinstance(offspring2, ArchitectureConfig)
        assert offspring1.input_size == 100
        assert offspring2.output_size == 10


class TestPerformancePredictor:
    """Test PerformancePredictor"""
    
    def test_create_predictor(self):
        """Test creating predictor"""
        predictor = PerformancePredictor()
        
        assert len(predictor.history) == 0
    
    def test_predict_performance(self):
        """Test predicting architecture performance"""
        predictor = PerformancePredictor()
        space = SearchSpace(input_size=100, output_size=10)
        
        config = space.sample_architecture()
        performance = predictor.predict(config)
        
        assert isinstance(performance, ArchitecturePerformance)
        assert 0 <= performance.accuracy <= 1.0
        assert performance.latency > 0
        assert performance.params > 0
    
    def test_predict_caching(self):
        """Test prediction caching"""
        predictor = PerformancePredictor()
        space = SearchSpace(input_size=100, output_size=10)
        
        config = space.sample_architecture()
        
        perf1 = predictor.predict(config)
        perf2 = predictor.predict(config)
        
        # Should return same cached result
        assert perf1.accuracy == perf2.accuracy
        assert perf1.params == perf2.params
    
    def test_update_with_actual(self):
        """Test updating with actual performance"""
        predictor = PerformancePredictor()
        space = SearchSpace(input_size=100, output_size=10)
        
        config = space.sample_architecture()
        actual_perf = ArchitecturePerformance(
            accuracy=0.96,
            latency=8.0,
            params=2000000,
            flops=10000000,
            memory=20.0,
            training_time=200.0
        )
        
        predictor.update_with_actual(config, actual_perf)
        
        # Next prediction should return actual
        predicted = predictor.predict(config)
        assert predicted.accuracy == 0.96


class TestNeuralArchitectureSearch:
    """Test NeuralArchitectureSearch"""
    
    def test_create_nas_random(self):
        """Test creating NAS with random strategy"""
        space = SearchSpace(input_size=100, output_size=10)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        assert nas.strategy == SearchStrategy.RANDOM
        assert len(nas.best_architectures) == 0
    
    def test_create_nas_evolutionary(self):
        """Test creating NAS with evolutionary strategy"""
        space = SearchSpace(input_size=100, output_size=10)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.EVOLUTIONARY,
            population_size=10
        )
        
        assert nas.strategy == SearchStrategy.EVOLUTIONARY
        assert nas.population_size == 10
    
    def test_random_search(self):
        """Test random search strategy"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        best_config, best_perf = nas.search(max_evaluations=10)
        
        assert isinstance(best_config, ArchitectureConfig)
        assert isinstance(best_perf, ArchitecturePerformance)
        assert len(nas.best_architectures) == 1
        assert len(nas.search_history) == 10
    
    def test_evolutionary_search(self):
        """Test evolutionary search strategy"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.EVOLUTIONARY,
            population_size=10,
            num_generations=3
        )
        
        best_config, best_perf = nas.search(max_evaluations=30)
        
        assert isinstance(best_config, ArchitectureConfig)
        assert isinstance(best_perf, ArchitecturePerformance)
        assert len(nas.best_architectures) == 1
    
    def test_rl_search(self):
        """Test RL-based search strategy"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.REINFORCEMENT_LEARNING
        )
        
        best_config, best_perf = nas.search(max_evaluations=15)
        
        assert isinstance(best_config, ArchitectureConfig)
        assert isinstance(best_perf, ArchitecturePerformance)
    
    def test_search_with_custom_weights(self):
        """Test search with custom performance weights"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        weights = {
            'accuracy': 0.8,
            'latency': 0.1,
            'params': 0.05,
            'memory': 0.05
        }
        
        best_config, best_perf = nas.search(
            max_evaluations=10,
            performance_weights=weights
        )
        
        assert isinstance(best_config, ArchitectureConfig)
    
    def test_get_top_k_architectures(self):
        """Test getting top k architectures"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        # Run multiple searches
        for _ in range(3):
            nas.search(max_evaluations=5)
        
        top_k = nas.get_top_k_architectures(k=2)
        
        assert len(top_k) <= 2
        # Check sorted by score
        if len(top_k) == 2:
            score1 = top_k[0][1].score()
            score2 = top_k[1][1].score()
            assert score1 >= score2
    
    def test_export_search_results(self):
        """Test exporting search results"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        nas.search(max_evaluations=5)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            nas.export_search_results(filepath)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(filepath)
            with open(filepath) as f:
                results = json.load(f)
            
            assert 'strategy' in results
            assert 'best_architectures' in results
            assert 'search_history' in results
            assert results['strategy'] == 'random'
        finally:
            os.unlink(filepath)
    
    def test_search_convergence(self):
        """Test that search improves over time"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.EVOLUTIONARY,
            population_size=10
        )
        
        best_config, best_perf = nas.search(max_evaluations=50)
        
        # Check that we have history
        assert len(nas.search_history) > 0
        
        # Best architecture should have reasonable performance
        assert best_perf.accuracy > 0.5
        assert best_perf.params > 0
    
    def test_multiple_strategies_comparison(self):
        """Test comparing different search strategies"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=5)
        
        results = {}
        for strategy in [SearchStrategy.RANDOM, SearchStrategy.EVOLUTIONARY]:
            nas = NeuralArchitectureSearch(
                search_space=space,
                strategy=strategy,
                population_size=10
            )
            
            best_config, best_perf = nas.search(max_evaluations=20)
            results[strategy.value] = best_perf.score()
        
        # All strategies should produce valid results
        assert all(score > 0 for score in results.values())


class TestIntegration:
    """Integration tests"""
    
    def test_full_nas_pipeline(self):
        """Test complete NAS pipeline"""
        # 1. Define search space
        space = SearchSpace(
            input_size=100,
            output_size=10,
            max_layers=8,
            min_layers=2,
            hidden_size_range=(32, 256)
        )
        
        # 2. Initialize NAS
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.EVOLUTIONARY,
            population_size=15,
            mutation_rate=0.15,
            crossover_rate=0.8
        )
        
        # 3. Run search
        best_config, best_perf = nas.search(max_evaluations=30)
        
        # 4. Get top architectures
        top_5 = nas.get_top_k_architectures(k=5)
        
        # 5. Export results
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            nas.export_search_results(filepath)
            
            # Verify
            assert isinstance(best_config, ArchitectureConfig)
            assert isinstance(best_perf, ArchitecturePerformance)
            assert len(top_5) <= 5
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)
    
    def test_nas_with_constraints(self):
        """Test NAS with performance constraints"""
        space = SearchSpace(input_size=100, output_size=10, max_layers=6)
        nas = NeuralArchitectureSearch(
            search_space=space,
            strategy=SearchStrategy.RANDOM
        )
        
        # Search with latency constraint
        weights = {
            'accuracy': 0.5,
            'latency': 0.4,  # High weight on latency
            'params': 0.05,
            'memory': 0.05
        }
        
        best_config, best_perf = nas.search(
            max_evaluations=20,
            performance_weights=weights
        )
        
        # Should find architecture balancing accuracy and latency
        assert best_perf.latency > 0
        assert best_perf.accuracy > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
