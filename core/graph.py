import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langgraph.graph import StateGraph, END
from core.state import KatalogState
from core.schemas import ProductContext, BrandRules
from agents.optimizer_agent import optimizer_agent
from agents.critic_agent import critic_agent

load_dotenv()

# Inicializamos Supabase en Modo Dios (Service Role)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# ==========================================
# 🛑 NODO 1: EXTRACCIÓN DE DATOS
# ==========================================
async def fetch_db_data(state: KatalogState):
    print(f"🔍 [Nodo 1] Buscando producto ID: {state['product_id']}")
    try:
        prod_res = supabase.table('shopify_products').select('*').eq('id', state['product_id']).single().execute()
        product_data = prod_res.data
        
        rules_res = supabase.table('brand_rules').select('*').eq('user_id', product_data['user_id']).single().execute()
        rules_data = rules_res.data

        context = ProductContext(
            shopify_id=product_data.get('shopify_id', ''),
            current_title=product_data.get('current_title', ''),
            current_body_html=product_data.get('current_body_html', ''),
            inventory_quantity=product_data.get('inventory_quantity', 0),
            sales_last_7_days=product_data.get('sales_last_7_days', 0)
        )
        
        rules = BrandRules(
            tone_voice=rules_data.get('tone_voice', 'Professional'),
            target_audience=rules_data.get('target_audience', 'General'),
            language=rules_data.get('language', 'English'),
            forbidden_words=rules_data.get('forbidden_words',[]),
            brand_dna=rules_data.get('brand_dna', ''),
            formatting_rules=rules_data.get('formatting_rules', '')
        )
        return {"product_context": context, "brand_rules": rules}
    except Exception as e:
        print(f"❌[Nodo 1] Error en DB: {e}")
        return {"error": str(e)}

# ==========================================
# 🧠 NODO 2: RECUPERAR MEMORIA
# ==========================================
async def retrieve_memory_letta(state: KatalogState):
    print("🧠 [Nodo 2] Consultando memoria a largo plazo en Letta...")
    return {"letta_memory": "Insight: Focus on benefits rather than just features."}

# ==========================================
# ✍️ NODO 3: LA IA ESCRIBE
# ==========================================
async def audit_and_write_pydantic(state: KatalogState):
    iteration = state.get("iterations", 0)
    print(f"✍️ [Nodo 3] Gemini Escribiendo (Intento {iteration + 1})...")
    
    if state.get("error"): return state 
    
    context = state["product_context"]
    rules = state["brand_rules"]
    memory = state["letta_memory"]
    feedback = state.get("critic_feedback")

    prompt = f"""
    PRODUCT TO OPTIMIZE:
    - Title: {context.current_title}
    - HTML: {context.current_body_html}
    - Inventory: {context.inventory_quantity}
    - Sales (7 days): {context.sales_last_7_days}

    BRAND RULES:
    - Tone: {rules.tone_voice}
    - Audience: {rules.target_audience}
    - Language: {rules.language}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - DNA: {rules.brand_dna}
    - Formats: {rules.formatting_rules}
    """

    # Si hay crítica previa, la inyectamos para que corrija
    if feedback:
        is_perf = feedback.get("is_perfect") if isinstance(feedback, dict) else getattr(feedback, "is_perfect", False)
        issues = feedback.get("issues_found") if isinstance(feedback, dict) else getattr(feedback, "issues_found",[])
        
        if not is_perf:
            prompt += f"\n⚠️ URGENT CRITIQUE: Fix these issues immediately: {issues}"

    try:
        result = await optimizer_agent.run(prompt)
        final_data = getattr(result, 'data', getattr(result, 'output', result))
        return {"final_proposal": final_data}
    except Exception as e:
        print(f"❌ [Nodo 3] Error de IA: {e}")
        return {"error": str(e)}

# ==========================================
# ⚖️ NODO 4: EL JUEZ
# ==========================================
async def review_proposal(state: KatalogState):
    iteration = state.get("iterations", 0) + 1
    print(f"⚖️ [Nodo 4] Juez Auditando propuesta (Intento {iteration})...")
    
    proposal = state.get("final_proposal")
    if state.get("error") or not proposal: 
        print("⚠️ [Nodo 4] No hay propuesta para evaluar.")
        return {"iterations": iteration}

    context = state["product_context"]
    rules = state["brand_rules"]

    prompt = f"""
    BRAND RULES TO ENFORCE:
    - Tone: {rules.tone_voice}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - Brand DNA: {rules.brand_dna}

    AI PROPOSAL TO REVIEW:
    {proposal}

    Verify if the proposal follows all rules strictly. 
    If it uses ANY forbidden words, reject it (is_perfect=False) and list them.
    If the title is over 70 chars, reject it.
    Provide actionable feedback for the writer to fix it.
    """

    try:
        result = await critic_agent.run(prompt)
        
        # TELEMETRÍA DE INGENIERÍA
        print(f"🔎[DEBUG JUEZ 1] Objeto devuelto: {type(result)}")
        
        # 🔥 EXTRACCIÓN BLINDADA (Buscamos donde sea que Pydantic esconda el dato)
        feed_data = getattr(result, 'data', getattr(result, 'output', result))
        
        print(f"🔎 [DEBUG JUEZ 2] Datos validados: {feed_data}")
        
        return {"critic_feedback": feed_data, "iterations": iteration}
    except Exception as e:
        print(f"❌ [Nodo 4] Error del Juez: {e}")
        return {"error": str(e), "iterations": iteration}

# ==========================================
# 🔀 ENRUTADOR (Lógica de Decisión Autónoma)
# ==========================================
def should_continue(state: KatalogState):
    if state.get("error"):
        return "save_db"
        
    feedback = state.get("critic_feedback")
    iterations = state.get("iterations", 0)
    
    print(f"🚦 [Enrutador] Analizando veredicto del Juez (Intento {iterations})...")
    
    # Comprobación estricta de None
    if feedback is None:
        print("⚠️[Decisión] El Juez no devolvió datos legibles. Guardando por seguridad...")
        return "save_db"
        
    # Extracción blindada (Soporta Dicts y Objetos Pydantic)
    is_perf = getattr(feedback, "is_perfect", False) if not isinstance(feedback, dict) else feedback.get("is_perfect", False)
    issues = getattr(feedback, "issues_found",["Errores desconocidos"]) if not isinstance(feedback, dict) else feedback.get("issues_found",["Errores desconocidos"])
    
    if is_perf:
        print("🟢 [Decisión] Calidad PERFECTA. Aprobado por el Juez.")
        return "save_db"
    elif iterations >= 3:
        print(f"🟠 [Decisión] Límite de 3 intentos alcanzado. Guardando mejor esfuerzo...")
        return "save_db"
    else:
        print(f"🔴[Decisión] Errores detectados: {issues}. Devolviendo al Escritor...")
        return "ai_writer"

# ==========================================
# 💾 NODO 5: GUARDAR EN BASE DE DATOS
# ==========================================
async def save_to_supabase(state: KatalogState):
    print("💾 [Nodo 5] Guardando resultados en Supabase...")
    proposal = state.get("final_proposal")
    
    if state.get("error") or not proposal: 
        print("⚠️ [Nodo 5] Nada que guardar.")
        return state

    try:
        proposal_dict = proposal if isinstance(proposal, dict) else (proposal.model_dump() if hasattr(proposal, 'model_dump') else str(proposal))
        score = proposal_dict.get('audit_score', 80) if isinstance(proposal_dict, dict) else 80
        
        supabase.table('shopify_products').update({
            'ai_proposal': proposal_dict,
            'audit_score': score,
            'audit_status': 'NEEDS_REVIEW' 
        }).eq('id', state['product_id']).execute()
        
        print("✅ [Nodo 5] Producto actualizado exitosamente.")
        return {"status": "SUCCESS"}
    except Exception as e:
        print(f"❌[Nodo 5] Error al guardar: {e}")
        return {"error": str(e)}

def build_graph():
    workflow = StateGraph(KatalogState)

    workflow.add_node("fetch_data", fetch_db_data)
    workflow.add_node("memory", retrieve_memory_letta)
    workflow.add_node("ai_writer", audit_and_write_pydantic)
    workflow.add_node("critic", review_proposal)
    workflow.add_node("save_db", save_to_supabase)

    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "memory")
    workflow.add_edge("memory", "ai_writer")
    workflow.add_edge("ai_writer", "critic") 

    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "save_db": "save_db",
            "ai_writer": "ai_writer"
        }
    )
    
    workflow.add_edge("save_db", END)
    return workflow.compile()

katalog_agent = build_graph()