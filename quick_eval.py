#!/usr/bin/env python3
"""
Quick Start: RAG vs Graph+LLM Evaluation with LLM Judge
Minimal example to get you started in 2 minutes
"""

import sys
import json

def quick_demo():
    """Run a minimal demo evaluation"""
    
    print("\n" + "="*80)
    print("RAG vs Graph+LLM Evaluation - Quick Demo")
    print("="*80 + "\n")
    
    # Check imports
    print("📦 Checking dependencies...")
    try:
        from sentence_transformers import SentenceTransformer
        from arango import ArangoClient
        import requests
        print("✅ All required packages available\n")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install with: pip install sentence-transformers arango requests")
        return
    
    # Check services
    print("🔍 Checking services...")
    
    # Check ArangoDB
    try:
        response = requests.get("http://localhost:8529", 
                              auth=("root", "openSesame"),
                              timeout=2)
        print("✅ ArangoDB running on localhost:8529")
    except:
        print("❌ ArangoDB not running on localhost:8529")
        print("   Start it with: docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=openSesame arangodb")
        return
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = response.json()
        if models.get('models'):
            print(f"✅ Ollama running on localhost:11434 with {len(models['models'])} model(s)")
        else:
            print("⚠️  Ollama running but no models found")
            print("   Pull a model with: ollama pull llama3.1:8b")
            return
    except:
        print("❌ Ollama not running on localhost:11434")
        print("   Start it with: ollama serve &")
        print("   Then pull model: ollama pull llama3.1:8b")
        return
    
    print("\n✅ All systems ready!\n")
    
    # Offer options
    print("Choose an option:")
    print("1. Run minimal test (1 query, ~2 minutes)")
    print("2. Run batch test (5 queries, ~10 minutes)")
    print("3. Run full evaluation framework")
    print("4. Just show documentation")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        run_minimal_test()
    elif choice == "2":
        run_batch_test()
    elif choice == "3":
        run_full_framework()
    elif choice == "4":
        show_documentation()
    else:
        print("Exiting.")

def run_minimal_test():
    """Run a single query test"""
    print("\n" + "="*80)
    print("Running Minimal Test (1 Query)")
    print("="*80 + "\n")
    
    try:
        from mitre_integrated_evaluation import IntegratedLLMEvaluator
        
        evaluator = IntegratedLLMEvaluator()
        
        query = "What techniques do threat actors use for credential theft?"
        print(f"\n📝 Query: {query}\n")
        
        result = evaluator.evaluate_single_query(query, show_responses=True)
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\nRAG Score:       {result['rag']['overall']:.1f}/10")
        print(f"  Relevance:     {result['rag']['relevance']:.1f}")
        print(f"  Completeness:  {result['rag']['completeness']:.1f}")
        print(f"  Accuracy:      {result['rag']['accuracy']:.1f}")
        print(f"  Specificity:   {result['rag']['specificity']:.1f}")
        print(f"  Clarity:       {result['rag']['clarity']:.1f}")
        
        print(f"\nGraph+LLM Score: {result['graph']['overall']:.1f}/10")
        print(f"  Relevance:     {result['graph']['relevance']:.1f}")
        print(f"  Completeness:  {result['graph']['completeness']:.1f}")
        print(f"  Accuracy:      {result['graph']['accuracy']:.1f}")
        print(f"  Specificity:   {result['graph']['specificity']:.1f}")
        print(f"  Clarity:       {result['graph']['clarity']:.1f}")
        
        print(f"\n{'='*80}")
        print(f"Winner: {result['comparison'].get('winner', 'TBD').upper()}")
        print(f"Improvement: {result['graph']['overall'] - result['rag']['overall']:+.1f} points")
        print(f"Reason: {result['comparison'].get('primary_reason', 'N/A')}")
        print(f"{'='*80}\n")
        
        # Save
        evaluator.save_results('/home/vasanthiyer-gpu/quick_test_results.json')
        print("✅ Results saved to quick_test_results.json\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_batch_test():
    """Run multiple queries"""
    print("\n" + "="*80)
    print("Running Batch Test (5 Queries)")
    print("="*80 + "\n")
    
    try:
        from mitre_integrated_evaluation import IntegratedLLMEvaluator
        
        evaluator = IntegratedLLMEvaluator()
        
        queries = [
            "What techniques do threat actors use for credential theft?",
            "How can we detect lateral movement?",
            "What are persistence mechanisms?",
            "What tools does APT28 use?",
            "How do we mitigate ransomware?"
        ]
        
        evaluator.evaluate_batch(queries, show_responses=False)
        
        # Generate report
        report = evaluator.generate_report(include_full_results=False)
        print(report)
        
        # Save
        evaluator.save_results('/home/vasanthiyer-gpu/batch_test_results.json')
        print("\n✅ Results saved to batch_test_results.json\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_full_framework():
    """Run the full integrated evaluation"""
    print("\n" + "="*80)
    print("Running Full Evaluation Framework")
    print("="*80 + "\n")
    
    print("Launching: python mitre_integrated_evaluation.py\n")
    import os
    os.system("python /home/vasanthiyer-gpu/mitre_integrated_evaluation.py")

def show_documentation():
    """Show available documentation"""
    print("\n" + "="*80)
    print("Available Documentation")
    print("="*80 + "\n")
    
    docs = {
        "EVALUATION_TOOLKIT_README.md": "Complete overview of all tools (START HERE)",
        "LLM_JUDGE_GUIDE.md": "Detailed guide to using the LLM judge",
        "RAG_VS_GRAPH_QUICK_START.md": "5-minute quick reference",
        "RAG_VS_GRAPH_EVALUATION_GUIDE.md": "Detailed metrics and manual scoring rubrics",
    }
    
    print("\n📚 Documentation Files:\n")
    for file, desc in docs.items():
        print(f"  • {file}")
        print(f"    {desc}\n")
    
    print("\n🐍 Python Scripts:\n")
    print("  • mitre_integrated_evaluation.py ⭐ START HERE")
    print("    Complete pipeline with LLM judge\n")
    
    print("  • mitre_llm_judge.py")
    print("    LLM judge component (standalone)\n")
    
    print("  • mitre_rag_vs_graph_comparison.py")
    print("    Response generation without judging\n")
    
    print("  • mitre_automated_metrics.py")
    print("    BLEU, ROUGE, hallucination metrics\n")
    
    print("\n📖 Read Documentation:\n")
    print("  cat EVALUATION_TOOLKIT_README.md")
    print("  cat LLM_JUDGE_GUIDE.md\n")

def main():
    """Main entry point"""
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
