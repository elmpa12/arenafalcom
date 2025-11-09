#!/usr/bin/env python3
"""
test_memory_integration.py
Testa integração completa: dialogue_engine + agent_memory
Simula dois diálogos em sequência para validar memória persistente
"""

import sys
import json
from pathlib import Path
from agent_memory import AgentMemory
from dialogue_engine import DialogueEngine

def print_section(title):
    """Imprime seção formatada"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_memory_structure():
    """Testa se estrutura de memória foi criada"""
    print_section("TEST 1: Verificando Estrutura de Memória")
    
    memory_dir = Path("/opt/botscalpv3/memory_store")
    
    if not memory_dir.exists():
        print("❌ Diretório memory_store não existe!")
        return False
    
    # Verifica estrutura esperada
    expected_dirs = [
        "Claude/dialogues",
        "Claude/specs",
        "Claude/decisions",
        "Claude/preferences",
        "Claude/relationships",
        "Codex/dialogues",
        "Codex/specs",
        "Codex/decisions",
        "Codex/preferences",
        "Codex/relationships",
        "shared"
    ]
    
    for subdir in expected_dirs:
        full_path = memory_dir / subdir
        if full_path.exists():
            print(f"  ✅ {subdir}/")
        else:
            print(f"  ❌ {subdir}/ FALTA")
            return False
    
    print("\n✅ Estrutura de memória está completa!")
    return True

def test_agent_memory_initialization():
    """Testa inicialização de AgentMemory"""
    print_section("TEST 2: Inicializando AgentMemory")
    
    try:
        memory_dir = Path("/opt/botscalpv3/memory_store")
        
        claude_mem = AgentMemory("Claude", str(memory_dir))
        print("✅ Claude memory inicializada")
        print(f"   Profile: {claude_mem.profile}")
        
        codex_mem = AgentMemory("Codex", str(memory_dir))
        print("✅ Codex memory inicializada")
        print(f"   Profile: {codex_mem.profile}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_memory_recording():
    """Testa gravação de dados na memória"""
    print_section("TEST 3: Gravando dados na memória")
    
    try:
        memory_dir = Path("/opt/botscalpv3/memory_store")
        claude_mem = AgentMemory("Claude", str(memory_dir))
        codex_mem = AgentMemory("Codex", str(memory_dir))
        
        # Simula um diálogo
        test_dialogue_id = "test_dialogue_20250308_001"
        test_dialogue = {
            "dialogue_id": test_dialogue_id,
            "requirement": "Test requirement",
            "consensus_reached": True,
            "rounds": 3,
            "exchange": [
                {"round": 1, "speaker": "Claude", "message": "Test message 1"},
                {"round": 2, "speaker": "Codex", "message": "Test message 2"},
                {"round": 3, "speaker": "Claude", "message": "Test message 3"}
            ]
        }
        
        # Grava diálogo
        claude_mem.record_dialogue(test_dialogue_id, test_dialogue)
        print("✅ Diálogo gravado para Claude")
        
        codex_mem.record_dialogue(test_dialogue_id, test_dialogue)
        print("✅ Diálogo gravado para Codex")
        
        # Testa gravação de preferência
        claude_mem.record_preference("architecture", "elegance_over_complexity", 9)
        print("✅ Preferência gravada para Claude")
        
        codex_mem.record_preference("performance", "speed_over_simplicity", 8)
        print("✅ Preferência gravada para Codex")
        
        # Testa gravação de relacionamento
        claude_mem.record_relationship("Codex", {
            "interaction": "collaborative",
            "agreement_level": 0.85,
            "notes": "Excellent technical synergy"
        })
        print("✅ Relacionamento gravado para Claude")
        
        codex_mem.record_relationship("Claude", {
            "interaction": "collaborative",
            "agreement_level": 0.85,
            "notes": "Complements my technical focus"
        })
        print("✅ Relacionamento gravado para Codex")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_retrieval():
    """Testa recuperação de contexto da memória"""
    print_section("TEST 4: Recuperando contexto da memória")
    
    try:
        memory_dir = Path("/opt/botscalpv3/memory_store")
        claude_mem = AgentMemory("Claude", str(memory_dir))
        codex_mem = AgentMemory("Codex", str(memory_dir))
        
        # Tenta recuperar contexto
        claude_context = claude_mem.get_context_for_dialogue()
        if claude_context:
            print("✅ Claude context recuperado:")
            print(f"   {claude_context[:150]}...")
        else:
            print("⚠️  Claude context vazio (primeira sessão?)")
        
        codex_context = codex_mem.get_context_for_dialogue()
        if codex_context:
            print("✅ Codex context recuperado:")
            print(f"   {codex_context[:150]}...")
        else:
            print("⚠️  Codex context vazio (primeira sessão?)")
        
        # Tenta recuperar conhecimento compartilhado
        shared_context = codex_mem.get_shared_context()
        if shared_context:
            print("✅ Shared knowledge recuperado:")
            print(f"   {shared_context[:150]}...")
        else:
            print("⚠️  Shared knowledge vazio")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dialogue_engine_with_memory():
    """Testa se DialogueEngine pode inicializar com memória"""
    print_section("TEST 5: Inicializando DialogueEngine com memória")
    
    try:
        engine = DialogueEngine(max_rounds=2)
        
        if engine.claude_memory:
            print("✅ DialogueEngine carregou memória de Claude")
        else:
            print("⚠️  DialogueEngine sem memória de Claude")
        
        if engine.codex_memory:
            print("✅ DialogueEngine carregou memória de Codex")
        else:
            print("⚠️  DialogueEngine sem memória de Codex")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_files_exist():
    """Verifica se arquivos de memória foram criados"""
    print_section("TEST 6: Verificando arquivos de memória criados")
    
    memory_dir = Path("/opt/botscalpv3/memory_store")
    
    # Procura por JSONL de diálogos
    claude_history = memory_dir / "Claude" / "dialogues" / "history.jsonl"
    codex_history = memory_dir / "Codex" / "dialogues" / "history.jsonl"
    
    if claude_history.exists():
        print(f"✅ Claude dialogue history: {claude_history}")
        with open(claude_history) as f:
            lines = f.readlines()
            print(f"   {len(lines)} entradas registradas")
    else:
        print(f"⚠️  Claude dialogue history ainda não existe")
    
    if codex_history.exists():
        print(f"✅ Codex dialogue history: {codex_history}")
        with open(codex_history) as f:
            lines = f.readlines()
            print(f"   {len(lines)} entradas registradas")
    else:
        print(f"⚠️  Codex dialogue history ainda não existe")
    
    # Procura por preferências
    claude_prefs = memory_dir / "Claude" / "preferences" / "index.json"
    codex_prefs = memory_dir / "Codex" / "preferences" / "index.json"
    
    if claude_prefs.exists():
        print(f"✅ Claude preferences: {claude_prefs}")
    
    if codex_prefs.exists():
        print(f"✅ Codex preferences: {codex_prefs}")
    
    return True

def main():
    """Executa todos os testes"""
    print("\n")
    print(" "*80)
    print("  🧪 MEMORY INTEGRATION TEST SUITE")
    print(" "*80)
    
    tests = [
        test_memory_structure,
        test_agent_memory_initialization,
        test_memory_recording,
        test_memory_retrieval,
        test_dialogue_engine_with_memory,
        test_memory_files_exist
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Erro ao executar teste: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Resumo final
    print_section("📊 RESUMO DE TESTES")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✅ Testes passados: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("\n💡 Próximas etapas:")
        print("   1. Execute: flabs --dialogue \"seu requisito\"")
        print("   2. Os agentes vão lembrar de diálogos anteriores")
        print("   3. Prefere serão salvos para sessões futuras")
        return 0
    else:
        print(f"\n⚠️  {total - passed} testes falharam")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
