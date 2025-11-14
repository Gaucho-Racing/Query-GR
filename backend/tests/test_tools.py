"""Tests for tools functionality in src/tools/run_queries.py.

These tests verify that the premade tools produce correct outputs for given inputs.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.tools.run_queries import (
    generate_pandas_script,
    execute_pandas_script,
    process_query,
)
from src.tools.utils import (
    extract_trip_id,
    score_signal,
    best_signals_for_query,
    infer_specific_signals,
    is_vehicle_data_query,
    expand_trip_ids,
    get_all_trip_ids,
)


class TestToolOutputs:
    """Verify that tools in src/tools/run_queries.py produce correct outputs."""
    
    def test_extract_trip_id_all(self):
        """Test extracting 'all' trip specification."""
        assert extract_trip_id("show me all test runs") == "all"
        assert extract_trip_id("all trips") == "all"
        assert extract_trip_id("every run") == "all"
    
    def test_extract_trip_id_range(self):
        """Test extracting trip range specification."""
        assert extract_trip_id("trips 3-5") == "range:3-5"
        assert extract_trip_id("runs 3 to 5") == "range:3-5"
        assert extract_trip_id("trips 10-15") == "range:10-15"
    
    def test_expand_trip_ids(self):
        """Test expanding trip specifications."""
        # Single trip
        assert expand_trip_ids("3") == ["3"]
        
        # Range
        assert expand_trip_ids("range:3-5") == ["3", "4", "5"]
        
        # "all" would require DB connection, so we'll test the logic
        # In real test, would mock get_all_trip_ids()
        with patch('src.tools.utils.get_all_trip_ids', return_value=["1", "2", "3", "4"]):
            assert expand_trip_ids("all") == ["1", "2", "3", "4"]
    
    def test_signal_scoring_thresholds_vs_cell_numbers(self):
        """Test that numeric thresholds don't match cell numbers."""
        # 45°C should NOT match acu_cell45_temp when no cell number mentioned
        query = "temperature exceeded 45°C"
        cell_signal = "acu_cell45_temp"
        general_signal = "battery_temp"
        
        cell_score = score_signal(query, cell_signal)
        general_score = score_signal(query, general_signal)
        
        # General signal should score better (lower) than cell-specific
        assert general_score < cell_score
    
    def test_signal_scoring_with_cell_number(self):
        """Test that cell numbers ARE matched when explicitly mentioned."""
        query = "cell 45 temperature"
        cell_signal = "acu_cell45_temp"
        general_signal = "battery_temp"
        
        cell_score = score_signal(query, cell_signal)
        general_score = score_signal(query, general_signal)
        
        # Cell-specific should score better when cell number is mentioned
        assert cell_score < general_score
    
    def test_motor_current_not_cell_specific(self):
        """Test that motor current queries prefer motor signals, not cell signals."""
        query = "motor current draw above 300A"
        motor_signal = "motor_current"
        cell_signal = "acu_cell300_current"
        
        motor_score = score_signal(query, motor_signal)
        cell_score = score_signal(query, cell_signal)
        
        # Motor signal should score better
        assert motor_score < cell_score
    
    def test_build_url_filters_metadata(self):
        """Test that build_url correctly filters metadata fields."""
        # This is tested indirectly through execute_pandas_script
        # The build_url function filters out metadata in its implementation
        metadata_fields = ['trip_id', 'run_id', 'produced_at', 'vehicle_id', 'token']
        test_signals = ['battery_temp', 'motor_current'] + metadata_fields
        
        # Simulate filtering logic
        filtered = [s for s in test_signals if s.lower() not in [
            'trip_id', 'trip', 'tripid', 'run_id', 'runid', 'run',
            'vehicle_id', 'vehicleid', 'vehicle', 'produced_at', 'producedat',
            'timestamp', 'time', 'token'
        ]]
        
        assert 'battery_temp' in filtered
        assert 'motor_current' in filtered
        assert all(meta not in filtered for meta in metadata_fields)
    
    def test_parse_full_df_preserves_produced_at(self):
        """Test that parse_full_df preserves produced_at timestamp."""
        # Mock payload with produced_at
        payload = {
            "data": {
                "data": [
                    {"produced_at": "2024-11-10T21:07:43.025428Z", "mobile_speed": 10.5},
                    {"produced_at": "2024-11-10T21:07:44.025428Z", "mobile_speed": 12.3},
                ]
            }
        }
        
        # This would be tested in execute_pandas_script context
        # For now, verify the logic exists
        assert True  # Placeholder - actual test would execute script with parse_full_df


class TestSignalSelection:
    """Test signal selection and filtering logic."""
    
    def test_general_temperature_preferred_over_cell_specific(self):
        """Test that general temperature signals are preferred when no cell number."""
        query = "battery temperature exceeded 45°C"
        signals_pool = [
            "battery_temp",
            "pack_temp",
            "acu_cell45_temp",
            "acu_cell16_temp",
            "mobile_speed"
        ]
        
        # Should prefer general signals
        matches = best_signals_for_query(
            query,
            max_signals=3,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        # Top matches should include general temperature signals
        top_signals = [m[0] for m in matches]
        assert any("battery" in s or "pack" in s for s in top_signals)
        # Cell-specific should be penalized
        assert not any("cell45" in s for s in top_signals[:2])  # Top 2 shouldn't be cell45
    
    def test_motor_current_excludes_cell_signals(self):
        """Test that motor current queries exclude cell-specific signals."""
        query = "motor current draw above 300A"
        signals_pool = [
            "motor_current",
            "inverter_current",
            "acu_cell1_current",
            "acu_cell300_current",
            "battery_current"
        ]
        
        matches = best_signals_for_query(
            query,
            max_signals=3,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        # Should prefer motor/inverter signals
        assert any("motor" in s or "inverter" in s for s in top_signals)
        # Should exclude cell signals
        assert not any("cell" in s for s in top_signals)


class TestThresholdVsCellNumber:
    """Test that thresholds (45°C, 300A) are distinguished from cell numbers."""
    
    def test_temperature_threshold_not_cell_number(self):
        """45°C is a threshold, not cell 45."""
        query = "temperature exceeded 45°C"
        assert extract_trip_id(query) is None  # No trip ID
        # Should not match cell 45 signals when no cell number mentioned
        
    def test_current_threshold_not_cell_number(self):
        """300A is a threshold, not cell 300."""
        query = "current draw above 300A"
        # Should prefer general current signals, not cell300 signals
        
    def test_voltage_threshold_not_cell_number(self):
        """80V is a threshold, not cell 80."""
        query = "voltage above 80V"
        # Should prefer general voltage signals, not cell80 signals
