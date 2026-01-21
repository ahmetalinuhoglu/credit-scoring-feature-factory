"""
Tests for Pipeline Orchestrator

Tests PipelineOrchestrator for end-to-end ML pipeline execution.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

from src.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestratorInit:
    """Test suite for PipelineOrchestrator initialization."""
    
    def test_init_basic(self, full_config):
        """Test PipelineOrchestrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir
            )
            
            assert orchestrator is not None
            assert orchestrator.name == "PipelineOrchestrator"
    
    def test_init_with_custom_name(self, full_config):
        """Test PipelineOrchestrator with custom name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir,
                name="CustomOrchestrator"
            )
            
            assert orchestrator.name == "CustomOrchestrator"
    
    def test_init_with_sample_data(self, full_config):
        """Test PipelineOrchestrator with sample data mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir,
                use_sample_data=True
            )
            
            assert orchestrator.use_sample_data is True
    
    def test_validate(self, full_config):
        """Test validate method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir
            )
            
            assert orchestrator.validate() is True


class TestPipelineOrchestratorRun:
    """Test suite for run method."""
    
    def test_run_all_stages(self, full_config):
        """Test running all pipeline stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir,
                use_sample_data=True
            )
            
            with patch.object(orchestrator, '_run_stage') as mock_run:
                mock_run.return_value = {}
                
                results = orchestrator.run()
                
                assert isinstance(results, dict)
    
    def test_run_specific_stages(self, full_config):
        """Test running specific pipeline stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir
            )
            
            with patch.object(orchestrator, '_run_stage') as mock_run:
                mock_run.return_value = {}
                
                results = orchestrator.run(stages=['data', 'features'])
                
                # Should have called run_stage for each specified stage
                assert mock_run.call_count >= 1


class TestPipelineOrchestratorSaveCleanup:
    """Test suite for save and cleanup methods."""
    
    def test_cleanup(self, full_config):
        """Test cleanup method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(
                config=full_config,
                output_dir=temp_dir
            )
            
            # Cleanup should not raise
            orchestrator._cleanup()
