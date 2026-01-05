#!/bin/bash
cd "/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding"

echo "Monitoring notebook execution..."
while [ ! -f submission_multi_strategy.csv ]; do
    echo "$(date): Waiting for submission_multi_strategy.csv..."
    sleep 30
done

echo "✓ Submission file created!"
ls -lh submission_multi_strategy.csv

echo ""
echo "Starting Day 2 benchmarking..."
python3 execute_day2.py

echo ""
echo "✓ Day 2 Complete!"
