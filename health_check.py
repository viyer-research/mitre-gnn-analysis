#!/usr/bin/env python3
"""
System Health Check for RAG vs Graph+LLM Evaluation Framework
Verifies all dependencies and services are ready
"""

import sys
import subprocess
import json
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n" + "="*70)
    print("Checking Python Packages")
    print("="*70)
    
    required_packages = {
        'sentence_transformers': 'SentenceTransformer',
        'arango': 'ArangoDB connector',
        'requests': 'HTTP library',
        'numpy': 'Numerical library',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:<30} ({package})")
        except ImportError:
            print(f"❌ {name:<30} ({package})")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def check_database():
    """Check if ArangoDB is running"""
    print("\n" + "="*70)
    print("Checking ArangoDB")
    print("="*70)
    
    try:
        import requests
        response = requests.get(
            "http://localhost:8529",
            auth=("root", "openSesame"),
            timeout=3
        )
        
        if response.status_code in [200, 401]:
            print("✅ ArangoDB is running on localhost:8529")
            
            # Check if database exists
            try:
                from arango import ArangoClient
                client = ArangoClient(hosts='http://localhost:8529')
                db = client.db('MITRE2kg', username='root', password='openSesame')
                
                # Check collections
                collections = db.collections()
                print(f"✅ MITRE2kg database exists with {len(collections)} collections:")
                for coll in collections:
                    print(f"   - {coll['name']}")
                
                return True
            except Exception as e:
                print(f"⚠️  Cannot connect to MITRE2kg database: {e}")
                print("   Run: python mitre2kg_inspector.py")
                return False
        else:
            print(f"❌ ArangoDB returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ArangoDB is not running on localhost:8529")
        print("   Start it with:")
        print("   docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=openSesame arangodb")
        return False
    except Exception as e:
        print(f"❌ Error checking ArangoDB: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    print("\n" + "="*70)
    print("Checking Ollama/LLM")
    print("="*70)
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        
        if response.status_code == 200:
            models = response.json()
            model_list = models.get('models', [])
            
            if model_list:
                print(f"✅ Ollama is running on localhost:11434")
                print(f"✅ Available models ({len(model_list)}):")
                for model in model_list:
                    name = model.get('name', 'unknown')
                    print(f"   - {name}")
                
                # Check for llama3.1:8b
                has_llama = any('llama3.1' in m.get('name', '').lower() 
                              for m in model_list)
                if has_llama:
                    print("✅ llama3.1:8b is available")
                else:
                    print("⚠️  llama3.1:8b not found")
                    print("   Pull it with: ollama pull llama3.1:8b")
                
                return True
            else:
                print("❌ Ollama running but no models found")
                print("   Pull a model with: ollama pull llama3.1:8b")
                return False
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running on localhost:11434")
        print("   Start it with:")
        print("   ollama serve &")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_files():
    """Check if evaluation framework files exist"""
    print("\n" + "="*70)
    print("Checking Evaluation Framework Files")
    print("="*70)
    
    base_path = Path("/home/vasanthiyer-gpu")
    
    required_files = {
        'quick_eval.py': 'Interactive menu',
        'mitre_integrated_evaluation.py': 'Main pipeline',
        'mitre_llm_judge.py': 'LLM judge',
        'mitre_rag_vs_graph_comparison.py': 'Response generation',
        'mitre_automated_metrics.py': 'Metrics calculation',
        'EVALUATION_TOOLKIT_README.md': 'Overview documentation',
        'LLM_JUDGE_GUIDE.md': 'Judge guide',
        'RAG_VS_GRAPH_QUICK_START.md': 'Quick start',
        'FILE_INVENTORY.md': 'File listing',
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = base_path / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename:<45} ({size} bytes)")
        else:
            print(f"❌ {filename:<45} (MISSING)")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    return True

def check_embedding_model():
    """Check if embedding model is available"""
    print("\n" + "="*70)
    print("Checking Embedding Model")
    print("="*70)
    
    try:
        from sentence_transformers import SentenceTransformer
        print("⏳ Loading all-MiniLM-L6-v2...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding
        test_embedding = model.encode("test query")
        print(f"✅ Embedding model loaded successfully")
        print(f"   Model: all-MiniLM-L6-v2")
        print(f"   Dimension: {len(test_embedding)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return False

def generate_health_report(results):
    """Generate health check report"""
    print("\n" + "="*70)
    print("HEALTH CHECK SUMMARY")
    print("="*70)
    
    checks = {
        'Python Packages': results['packages'],
        'ArangoDB': results['database'],
        'Ollama/LLM': results['ollama'],
        'Embedding Model': results['embedding'],
        'Framework Files': results['files'],
    }
    
    all_ok = all(checks.values())
    
    print()
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check:<30} {'OK' if status else 'FAILED'}")
    
    print("\n" + "="*70)
    
    if all_ok:
        print("✅ ALL SYSTEMS READY!")
        print("\nYou can now run:")
        print("  python quick_eval.py")
        print("  python mitre_integrated_evaluation.py")
        return 0
    else:
        print("❌ Some checks failed. Fix the issues above and try again.")
        return 1

def main():
    """Main health check"""
    print("\n" + "="*70)
    print("RAG vs Graph+LLM Evaluation Framework - Health Check")
    print("="*70)
    
    results = {
        'packages': check_python_packages(),
        'database': check_database(),
        'ollama': check_ollama(),
        'embedding': check_embedding_model(),
        'files': check_files(),
    }
    
    exit_code = generate_health_report(results)
    
    # Save report
    report_file = Path("/home/vasanthiyer-gpu/health_check_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            'all_ok': all(results.values()),
            'checks': results,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📄 Health check report saved to: health_check_report.json\n")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
