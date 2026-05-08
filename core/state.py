from typing import TypedDict, Optional, Any
from core.schemas import ProductContext, BrandRules, AIProposalOutput, CriticFeedback

class KatalogState(TypedDict):
    """
    La memoria a corto plazo (RAM) de la ejecución actual.
    Ahora con 'critic_feedback' e 'iterations' para soportar el bucle de calidad.
    """
    product_id: str
    user_id: Optional[str]
    auto_pilot_enabled: bool
    product_context: Optional[ProductContext]
    brand_rules: Optional[BrandRules]
    
    # Memoria y Propuestas
    letta_memory: Optional[str] 
    final_proposal: Optional[AIProposalOutput]
    
    # 🛡️ CAPAS DE SEGURIDAD Y BUCLE DE CRÍTICA
    critic_feedback: Optional[Any] # Aquí vive el veredicto del Juez
    iterations: int               # Contador de seguridad (Max 3 intentos)
    
    # Manejo de errores global
    error: Optional[str]
