from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
from typing import Union, Optional
from core.graph import katalog_agent

# Inicializamos la API
app = FastAPI(
    title="Katalog AI Brain", 
    description="The Revenue Optimizer API - Powered by LangGraph & Gemini 3.1", 
    version="2.1"
)

# 🛡️ CORS (Configurado para permitir comunicación con el Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambiaremos esto por tu dominio de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELO DE DATOS BLINDADO (Pydantic v2) ---
class OptimizeRequest(BaseModel):
    """
    Estructura de entrada inteligente.
    - Union[str, int]: Acepta el ID tanto si viene como "6" o como 6.
    - AliasChoices: Mapea 'productId' (estándar de JS) o 'product_id' al mismo campo.
    """
    product_id: Union[str, int] = Field(
        ..., 
        validation_alias=AliasChoices("product_id", "productId")
    )

@app.post("/api/optimize")
async def optimize_product(request: OptimizeRequest):
    # Convertimos siempre a string para que LangGraph y Supabase no tengan errores de tipos
    clean_id = str(request.product_id)
    
    print(f"🚀 [API] Petición recibida: Optimizar producto ID {clean_id}")
    
    try:
        # 1. Definimos el estado inicial para el flujo de LangGraph
        initial_state = {"product_id": clean_id}
        
        # 2. INVOCAMOS A LANGGRAPH (Modo Asíncrono)
        # Esto dispara secuencialmente: Fetch -> Memory -> AI Writer -> Save
        final_state = await katalog_agent.ainvoke(initial_state)
        
        # 3. Verificamos si el grafo capturó algún error en sus nodos
        if final_state.get("error"):
            print(f"⚠️ [Grafo] Error detectado en el flujo: {final_state['error']}")
            raise HTTPException(status_code=500, detail=final_state["error"])
            
        return {
            "status": "SUCCESS", 
            "message": "Asset optimized successfully", 
            "data": final_state.get("final_proposal")
        }
        
    except Exception as e:
        print(f"❌ [API] Error Fatal en el proceso: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {
        "status": "ONLINE", 
        "brain": "Katalog AI 2.1",
        "engine": "LangGraph + Gemini 3.1 Pro"
    }