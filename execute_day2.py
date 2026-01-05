#!/usr/bin/env python3
"""
DAY 2 EXECUTION SCRIPT: Benchmark & Visualization
Runs performance evaluation on multi-strategy recall system
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Change to coding directory
os.chdir("/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding")

print("\n" + "="*80)
print("🚀 DAY 2: BENCHMARK & VISUALIZATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Check if required files exist
print("\n📋 Step 1: Verifying required files...")
required_files = [
    "benchmark_strategies.py",
    "visualize_system.py", 
    "multi_strategy_recall.py",
    "新闻推荐系统-多路召回.ipynb"
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n⚠️  Some required files are missing. Aborting.")
    sys.exit(1)

# Step 2: Create outputs directory
print("\n📁 Step 2: Setting up output directory...")
os.makedirs("outputs", exist_ok=True)
print("  ✅ Created/verified: outputs/")

# Step 3: Run benchmark
print("\n" + "="*80)
print("⏱️  PHASE 1: Running Benchmarks (Step by step)")
print("="*80)
print("This will evaluate:")
print("  - Recall@K (K=5, 10, 20, 50)")
print("  - NDCG@K")
print("  - Precision@K")
print("  - Coverage")
print("  - Latency per prediction")
print("\nEstimated time: 30-40 minutes")
print("\nStarting benchmark.py...")
print("-"*80)

benchmark_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "benchmark_strategies.py"],
        capture_output=False,
        timeout=2400  # 40 minute timeout
    )
    benchmark_time = time.time() - benchmark_start
    
    if result.returncode == 0:
        print("-"*80)
        print(f"✅ Benchmark completed successfully! ({benchmark_time/60:.1f} minutes)")
    else:
        print(f"⚠️  Benchmark finished with code {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("❌ Benchmark timed out after 40 minutes")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running benchmark: {e}")
    sys.exit(1)

# Step 4: Run visualizations
print("\n" + "="*80)
print("📊 PHASE 2: Running Visualizations")
print("="*80)
print("This will generate:")
print("  - User activity distribution")
print("  - Item popularity distribution")
print("  - Recommendation diversity analysis")
print("  - Strategy comparison charts")
print("  - Performance metrics visualization")
print("\nEstimated time: 15-25 minutes")
print("\nStarting visualize_system.py...")
print("-"*80)

viz_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "visualize_system.py"],
        capture_output=False,
        timeout=1500  # 25 minute timeout
    )
    viz_time = time.time() - viz_start
    
    if result.returncode == 0:
        print("-"*80)
        print(f"✅ Visualization completed successfully! ({viz_time/60:.1f} minutes)")
    else:
        print(f"⚠️  Visualization finished with code {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("❌ Visualization timed out after 25 minutes")
except Exception as e:
    print(f"⚠️  Error running visualization: {e}")
    print("    (This is usually non-critical for metrics collection)")

# Step 5: Summary
print("\n" + "="*80)
print("✅ DAY 2 EXECUTION COMPLETE")
print("="*80)

total_time = time.time() - benchmark_start
print(f"\nTotal execution time: {total_time/60:.1f} minutes")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📂 Generated files in outputs/:")
if os.path.exists("outputs"):
    output_files = os.listdir("outputs")
    if output_files:
        for i, file in enumerate(sorted(output_files), 1):
            filepath = os.path.join("outputs", file)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  {i}. {file} ({size:.1f} KB)")
    else:
        print("  (No files generated yet)")

print("\n📊 NEXT STEPS:")
print("  1. Review generated metrics in outputs/")
print("  2. Compare performance: Original ItemCF vs Multi-Strategy")
print("  3. Document improvement: 42% → 44.5% (or actual numbers)")
print("  4. Prepare for Day 3-4: CV & Interview Prep")
print("\n📖 Read: DAY2_BENCHMARK_AND_VISUALIZATION.md for detailed analysis")
print("\n" + "="*80)
