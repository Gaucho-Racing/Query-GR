#!/bin/bash

# Test runner with custom summary and failure reporting
cd "$(dirname "$0")"

echo "======================================================================"
echo "🧪 Running Tests"
echo "======================================================================"
echo ""
echo "📁 Test files:"
echo "   • tests/test_llm.py"
echo "   • tests/test_tools.py"
echo ""
echo "======================================================================"
echo ""

# Create temp file for capturing output
temp_file=$(mktemp)

# Run pytest with verbose output
PYTHONUNBUFFERED=1 python3 -u -m pytest tests/test_llm.py tests/test_tools.py -v --tb=short --color=yes 2>&1 | tee "$temp_file"
exit_code=${PIPESTATUS[0]}

echo ""
echo "======================================================================"
echo "📋 Test Summary"
echo "======================================================================"

# Extract summary from temp_file
summary=$(grep -E "[0-9]+ passed|[0-9]+ failed|[0-9]+ error" "$temp_file" | tail -1)

if [ -n "$summary" ]; then
    echo "$summary"
    echo ""
    
    # Extract numbers
    passed=$(echo "$summary" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
    failed=$(echo "$summary" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
    error=$(echo "$summary" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo "0")
    
    total=$((passed + failed + error))
    
    echo "Total: $total | ✅ Passed: $passed | ❌ Failed: $failed | ⚠️ Errors: $error"
    echo ""

    if [ "$failed" -gt 0 ] || [ "$error" -gt 0 ]; then
        echo "======================================================================"
        echo "❌ Failed Tests (with reasons):"
        echo "======================================================================"
        echo ""
        grep -A 10 "FAILED\|ERROR" "$temp_file" | grep -E "(FAILED|ERROR|AssertionError|assert)" | head -20
        echo ""
        echo "For full details, see output above."
    else
        echo "✅ All tests passed!"
    fi
else
    echo "⚠️  Could not parse test results"
    echo ""
    echo "Last 20 lines of output:"
    tail -20 "$temp_file"
fi

echo ""
echo "======================================================================"

# Clean up
rm -f "$temp_file"

exit $exit_code
