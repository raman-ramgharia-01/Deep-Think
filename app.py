# Test imports - run this in a separate cell or at the top
print("Testing imports...")

try:
    from rag_system import rag_system
    print("✅ rag_system imported")
    print(f"Type: {type(rag_system)}")
except Exception as e:
    print(f"❌ Failed to import rag_system: {e}")

try:
    from ResearchSystem import self_research
    print("✅ self_research imported")
    print(f"Type: {type(self_research)}")
except Exception as e:
    print(f"❌ Failed to import self_research: {e}")
