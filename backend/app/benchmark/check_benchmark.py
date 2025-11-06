# check_benchmark.py
import sys
import os
sys.path.append('.')

try:
    from app.benchmark.jee_bench_loader import JEEBenchLoader
    loader = JEEBenchLoader()
    print("✅ JEEBenchLoader imported successfully")
    
    # Check what's inside
    print("Loader methods:", [method for method in dir(loader) if not method.startswith('_')])
    
except Exception as e:
    print(f"❌ Import error: {e}")