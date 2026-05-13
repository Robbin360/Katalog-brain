from typing import Any, NotRequired, Required, TypedDict
from core.schemas import ProductContext, BrandRules, AIProposalOutput, CriticFeedback

class KatalogState(TypedDict, total=False):
    """
    La memoria a corto plazo (RAM) de la ejecución actual.
    Ahora con 'critic_feedback' e 'iterations' para soportar el bucle de calidad.
    """
    product_id: Required[str]
    user_id: NotRequired[str]
    auto_pilot_enabled: NotRequired[bool]
    product_context: NotRequired[ProductContext]
    brand_rules: NotRequired[BrandRules]
    
    # Memoria y Propuestas
    letta_memory: NotRequired[str]
    final_proposal: NotRequired[AIProposalOutput]
    
    # 🛡️ CAPAS DE SEGURIDAD Y BUCLE DE CRÍTICA
    critic_feedback: NotRequired[CriticFeedback | Any] # Aquí vive el veredicto del Juez
    iterations: NotRequired[int]                       # Contador de seguridad (Max 3 intentos)
    
    # Manejo de errores global
    error: NotRequired[str]
    status: NotRequired[str]

    # Metadatos de optimización
    framework_used: NotRequired[str]
    tone_used: NotRequired[str]
    description_length: NotRequired[int]
