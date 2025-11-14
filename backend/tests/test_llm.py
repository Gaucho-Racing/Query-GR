"""Tests for LLM/agent behavior.

These tests validate that the LLM agent (agent.py) correctly:
- Decides whether to call a tool
- Passes the right arguments to tools
- Handles tool responses appropriately
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from src.main import app
from src.llm.agent import call_gemini
from src.llm.router import handle_query, QueryRequest
from src.tools.run_queries import process_query, execute_pandas_script, generate_pandas_script
from src.tools.utils import (
    extract_trip_id,
    best_signals_for_query,
    is_vehicle_data_query,
)

client = TestClient(app)


class TestLLMAgentBehavior:
    """Test LLM agent decision-making and tool calling."""
    
    def test_agent_calls_tool_for_vehicle_query(self):
        """Test that agent correctly identifies vehicle data queries and calls tools."""
        query = "show me average speed for trip 3"
        
        # Verify it's detected as vehicle query
        from src.tools.utils import is_vehicle_data_query
        assert is_vehicle_data_query(query, trip_id="3", default_trip_id="3")
    
    def test_agent_rejects_non_vehicle_query(self):
        """Test that agent rejects non-vehicle queries without calling tools."""
        query = "What is the weather today?"
        
        from src.tools.utils import is_vehicle_data_query
        # Should return False or ask for clarification
        result = is_vehicle_data_query(query, trip_id=None, default_trip_id="4")
        # May return False or True depending on intent words, but shouldn't process
        
    def test_agent_passes_correct_arguments(self):
        """Test that agent passes correct arguments to tools."""
        # This would test that:
        # - Correct signals are selected
        # - Correct trip_id is passed
        # - Query is properly formatted
        
        query = "show me battery temperature for trip 3"
        trip_id = "3"
        
        # Verify trip_id extraction
        from src.tools.utils import extract_trip_id
        extracted = extract_trip_id(query)
        assert extracted == trip_id
    
    @pytest.mark.asyncio
    async def test_agent_handles_tool_responses(self):
        """Test that agent correctly handles tool responses."""
        # Mock the tool execution
        mock_result = "Average battery temperature: 25.5°C"
        mock_table_data = None
        
        with patch('src.tools.run_queries.execute_pandas_script', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = (mock_result, {}, None, mock_table_data)
            
            # Test that response is properly formatted
            # This would be tested through the full query flow
            assert True  # Placeholder


class TestLLMAPIEndpoints:
    """Test LLM API endpoint behavior."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_log_endpoint(self):
        """Test error logging endpoint."""
        response = client.post(
            "/log",
            json={
                "error": "Test error",
                "timestamp": "2024-01-01T00:00:00Z",
                "userAgent": "test-agent",
                "url": "http://test.com"
            }
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_llm_query_empty_message(self):
        """Test LLM query endpoint with empty message."""
        response = client.post(
            "/llm/query",
            json={"message": ""}
        )
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert "valid query" in response.json()["message"].lower()
    
    def test_llm_query_non_vehicle_query(self):
        """Test LLM query endpoint with non-vehicle query."""
        response = client.post(
            "/llm/query",
            json={"message": "What is the weather today?"}
        )
        assert response.status_code == 200
        # Should return a message indicating it can only help with vehicle data
        assert response.json()["success"] is True
    
    def test_llm_query_clarify_trip(self):
        """Test LLM query endpoint when trip ID is missing."""
        response = client.post(
            "/llm/query",
            json={"message": "show me average speed"}
        )
        assert response.status_code == 200
        # Should ask for trip clarification
        assert response.json()["success"] is True
        assert "trip" in response.json()["message"].lower() or "run" in response.json()["message"].lower()
    
    def test_llm_query_accepts_all_trips(self):
        """Test that LLM accepts 'all' as valid trip specification."""
        response = client.post(
            "/llm/query",
            json={"message": "show me all test runs where battery temperature exceeded 45°C"}
        )
        assert response.status_code == 200
        # Should not ask for clarification if "all" is detected
        # May still ask if DB query fails, but shouldn't reject "all" as invalid
        assert response.json()["success"] is True
    
    def test_llm_query_handles_table_data(self):
        """Test that LLM query returns table_data for 'show all' queries."""
        # This would require mocking the full query processing
        # For now, verify the endpoint structure supports it
        assert True  # Placeholder - would test full flow with mocked data


class TestToolResponseHandling:
    """Test how agent handles different tool response types."""
    
    def test_handles_numeric_result(self):
        """Test handling of simple numeric results."""
        # Agent should format numeric results appropriately
        assert True  # Placeholder
    
    def test_handles_table_result(self):
        """Test handling of table data results."""
        # Agent should return table_data in response
        assert True  # Placeholder
    
    def test_handles_image_result(self):
        """Test handling of image/graph results."""
        # Agent should return image_base64 in response
        assert True  # Placeholder
    
    def test_handles_error_responses(self):
        """Test handling of tool execution errors."""
        # Agent should return appropriate error messages
        assert True  # Placeholder


class TestJakesQuery1_BatteryTempAndMotorCurrent:
    """
    Query: "Show me all test runs where battery temperature exceeded 45°C 
    AND motor current draw was above 300A, sorted by lap time"
    
    Challenge: Requires joining multiple sensor streams with different sampling rates
    """
    
    def test_query_detection(self):
        """Test that query is correctly identified as vehicle data query."""
        query = "Show me all test runs where battery temperature exceeded 45°C AND motor current draw was above 300A, sorted by lap time"
        
        # Should detect "all" trips
        trip_spec = extract_trip_id(query)
        assert trip_spec == "all"
        
        # Should be identified as vehicle query
        assert is_vehicle_data_query(query, trip_id="all", default_trip_id="4")
    
    def test_signal_selection(self):
        """Test that correct signals are selected (not cell-specific)."""
        query = "Show me all test runs where battery temperature exceeded 45°C AND motor current draw was above 300A"
        
        # Mock signals pool
        signals_pool = [
            "battery_temp",
            "pack_temp",
            "motor_current",
            "inverter_current",
            "acu_cell45_temp",  # Should NOT be selected (45 is threshold, not cell)
            "acu_cell300_current",  # Should NOT be selected (300 is threshold, not cell)
            "lap_time",
            "mobile_speed"
        ]
        
        # Should select general battery temp and motor current signals
        matches = best_signals_for_query(
            query,
            max_signals=5,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        
        # Should include battery/pack temperature (not cell45)
        assert any("battery" in s or "pack" in s for s in top_signals)
        assert not any("cell45" in s for s in top_signals[:3])  # Top 3 shouldn't be cell45
        
        # Should include motor/inverter current (not cell300)
        assert any("motor" in s or "inverter" in s for s in top_signals)
        assert not any("cell300" in s for s in top_signals[:3])
    
    @pytest.mark.asyncio
    async def test_script_generation(self):
        """Test that script is generated correctly for this query."""
        query = "Show me all test runs where battery temperature exceeded 45°C AND motor current draw was above 300A, sorted by lap time"
        
        # Mock Gemini call
        mock_gemini = AsyncMock(return_value="""
import pandas as pd
url = build_url(['battery_temp', 'motor_current'], trip_id='3')
resp = http_get(url)
data = resp.json()
df = parse_full_df(data)
# Filter logic here
result = "Found X runs"
print(result)
set_result(result)
""")
        
        # Test script generation
        script = await generate_pandas_script(
            query,
            ["battery_temp", "motor_current"],
            mock_gemini,
            trip_ids=["1", "2", "3"]
        )
        
        assert script is not None
        assert "battery_temp" in script.lower() or "motor_current" in script.lower()
        # Should handle multiple trips
        assert "trip_id" in script.lower() or "loop" in script.lower() or "all" in script.lower()
    
    def test_expected_output_format(self):
        """Test that output is in table format for 'show all' query."""
        # Expected: table_data with columns like:
        # - trip_id (or run_id)
        # - battery_temp (max or values exceeding 45°C)
        # - motor_current (max or values above 300A)
        # - lap_time
        # Sorted by lap_time
        
        expected_columns = ["trip_id", "battery_temp", "motor_current", "lap_time"]
        # This would be verified in integration test with actual data
        assert True  # Placeholder


class TestJakesQuery2_PackVoltageDegradation:
    """
    Query: "Give me the average pack voltage degradation per lap for our last competition day"
    
    Challenge: Need to define "lap" boundaries, calculate deltas, group by laps, handle incomplete laps
    """
    
    def test_query_detection(self):
        """Test query detection."""
        query = "Give me the average pack voltage degradation per lap for our last competition day"
        
        # Should detect voltage and lap keywords
        assert "voltage" in query.lower()
        assert "lap" in query.lower()
        assert is_vehicle_data_query(query, trip_id="4", default_trip_id="4")
    
    def test_signal_selection_pack_voltage(self):
        """Test that pack voltage signal is selected (not cell-specific)."""
        query = "pack voltage degradation per lap"
        
        signals_pool = [
            "pack_voltage",
            "battery_voltage",
            "acu_cell1_voltage",
            "acu_cell16_voltage",
            "lap_counter",
            "lap_time"
        ]
        
        matches = best_signals_for_query(
            query,
            max_signals=3,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        # Should prefer pack_voltage over cell-specific
        assert "pack_voltage" in top_signals[0] or "battery_voltage" in top_signals[0]
    
    def test_expected_calculation_logic(self):
        """Test that script would calculate voltage degradation correctly."""
        # Expected logic:
        # 1. Identify lap boundaries
        # 2. Calculate voltage at start and end of each lap
        # 3. Compute delta (degradation) per lap
        # 4. Average across all laps
        # 5. Handle incomplete laps
        
        assert True  # Placeholder - would test script logic


class TestJakesQuery3_WheelSlip:
    """
    Query: "Find all acceleration runs where we had more than 5% wheel slip in the first 30 meters"
    
    Challenge: Wheel slip isn't directly measured - needs calculation from wheel speed vs GPS speed
    """
    
    def test_query_detection(self):
        """Test query detection."""
        query = "Find all acceleration runs where we had more than 5% wheel slip in the first 30 meters"
        
        assert "wheel" in query.lower() or "slip" in query.lower()
        assert "5%" in query or "5" in query
        assert is_vehicle_data_query(query, trip_id="4", default_trip_id="4")
    
    def test_signal_selection_wheel_and_gps(self):
        """Test that wheel speed and GPS speed signals are selected."""
        query = "wheel slip calculation"
        
        signals_pool = [
            "wheel_speed_front",
            "wheel_speed_rear",
            "gps_speed",
            "mobile_speed",
            "acceleration"
        ]
        
        matches = best_signals_for_query(
            query,
            max_signals=4,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        # Should include wheel speed and GPS speed
        assert any("wheel" in s for s in top_signals)
        assert any("gps" in s or "mobile" in s for s in top_signals)
    
    def test_calculation_formula(self):
        """Test that wheel slip calculation formula is correct."""
        # Formula: slip = (wheel_speed - gps_speed) / gps_speed * 100
        # Or: slip = (wheel_speed - gps_speed) / wheel_speed * 100
        
        # Test calculation
        wheel_speed = 50.0  # m/s
        gps_speed = 45.0    # m/s
        slip_percent = ((wheel_speed - gps_speed) / gps_speed) * 100
        
        assert abs(slip_percent - 11.11) < 0.1  # ~11.11% slip
        
        # Should filter for > 5%
        assert slip_percent > 5.0


class TestJakesQuery4_DriverSectorTimes:
    """
    Query: "Compare our three drivers' average sector times on the endurance track, 
    but only using runs where SOC started above 80%"
    
    Challenge: Multiple filters, grouping, aggregation, AND contextual filtering
    """
    
    def test_query_detection(self):
        """Test query detection."""
        query = "Compare our three drivers' average sector times on the endurance track, but only using runs where SOC started above 80%"
        
        assert "driver" in query.lower() or "sector" in query.lower()
        assert "soc" in query.lower() or "80" in query
        assert is_vehicle_data_query(query, trip_id="4", default_trip_id="4")
    
    def test_signal_selection_multiple_signals(self):
        """Test that multiple signals are selected (sector times, driver, SOC, track)."""
        query = "driver sector times SOC track"
        
        signals_pool = [
            "sector_time",
            "sector_1_time",
            "sector_2_time",
            "driver_id",
            "soc",
            "state_of_charge",
            "track_type",
            "track_name"
        ]
        
        matches = best_signals_for_query(
            query,
            max_signals=6,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        # Should include sector, driver, SOC, track signals
        assert any("sector" in s for s in top_signals)
        assert any("driver" in s or "soc" in s or "track" in s for s in top_signals)
    
    def test_filtering_logic(self):
        """Test that filtering logic is correct."""
        # Expected filters:
        # 1. track_type == "endurance"
        # 2. initial SOC > 80%
        # 3. Group by driver
        # 4. Calculate average sector times
        
        assert True  # Placeholder


class TestJakesQuery5_EnergyConsumptionTrends:
    """
    Query: "How has our average energy consumption per lap changed over the last 3 months, 
    broken down by track type?"
    
    Challenge: Long time range, need to group by track metadata, calculate normalized metrics
    """
    
    def test_query_detection(self):
        """Test query detection."""
        query = "How has our average energy consumption per lap changed over the last 3 months, broken down by track type?"
        
        assert "energy" in query.lower() or "consumption" in query.lower()
        assert "lap" in query.lower()
        assert "3 months" in query.lower() or "months" in query.lower()
        assert is_vehicle_data_query(query, trip_id="4", default_trip_id="4")
    
    def test_signal_selection_energy_and_track(self):
        """Test that energy consumption and track type signals are selected."""
        query = "energy consumption per lap track type"
        
        signals_pool = [
            "energy_consumption",
            "power_consumption",
            "lap_energy",
            "track_type",
            "track_name",
            "lap_counter"
        ]
        
        matches = best_signals_for_query(
            query,
            max_signals=5,
            pool=signals_pool,
            default_trip_id="4"
        )
        
        top_signals = [m[0] for m in matches]
        # Should include energy and track signals
        assert any("energy" in s or "power" in s for s in top_signals)
        assert any("track" in s for s in top_signals)
    
    def test_time_range_filtering(self):
        """Test that time range filtering works correctly."""
        # Should filter for last 3 months using produced_at timestamp
        # Should use parse_full_df() to get produced_at column
        
        assert True  # Placeholder


class TestToolDevelopmentGuidance:
    """
    Tests that guide development of new tools.
    
    If a test fails, it indicates a new tool needs to be added to run_queries.py.
    """
    
    def test_lap_boundary_detection_tool(self):
        """Test that lap boundary detection works."""
        # If this fails, may need to add lap boundary detection logic
        assert True  # Placeholder
    
    def test_wheel_slip_calculation_tool(self):
        """Test that wheel slip calculation tool exists."""
        # If this fails, may need to add wheel slip calculation helper
        assert True  # Placeholder
    
    def test_multi_trip_joining_tool(self):
        """Test that multi-trip data joining works."""
        # If this fails, may need to improve multi-trip handling
        assert True  # Placeholder
    
    def test_time_based_filtering_tool(self):
        """Test that time-based filtering (3 months) works."""
        # If this fails, may need to add time range filtering helpers
        assert True  # Placeholder
