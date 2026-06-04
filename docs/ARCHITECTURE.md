# Mapa del Sistema y Arquitectura: Katalog-Brain

Este documento está diseñado especialmente para Product Managers, CEOs y desarrolladores que migran desde herramientas No-Code visuales (como n8n) hacia **LangGraph**, facilitando la comprensión y edición del flujo de trabajo de Katalog-Brain.

---

## 📂 Estructura del Proyecto (Árbol de Carpetas)

El proyecto está organizado en tres directorios clave para separar la lógica de ejecución, el comportamiento de la IA y los scripts operativos:

*   **`core/`**: Es el **corazón del sistema**.
    *   [graph.py](file:///c:/proyectos/Katalog-brain/core/graph.py): Define la red de nodos, las conexiones lógicas condicionales y compila el flujo de ejecución (grafo).
    *   [state.py](file:///c:/proyectos/Katalog-brain/core/state.py): Especifica el "Estado" (`KatalogState`), que actúa como la memoria a corto plazo (RAM) compartida entre todos los nodos durante la optimización de un producto.
    *   [schemas.py](file:///c:/proyectos/Katalog-brain/core/schemas.py): Define los esquemas y validaciones de datos estrictas usando Pydantic (como las propuestas generadas por la IA y el feedback de la crítica).
    *   [shopify_api.py](file:///c:/proyectos/Katalog-brain/core/shopify_api.py): Módulo para interactuar con la API GraphQL de Shopify para consultar taxonomía predictiva.
    *   [shopify_tools.py](file:///c:/proyectos/Katalog-brain/core/shopify_tools.py): Herramientas para publicar los cambios optimizados de vuelta a las tiendas de Shopify.
*   **`agents/`**: Aloja los **cerebros cognitivos**. Cada archivo aquí define las directivas y prompts para un agente especializado de IA:
    *   [optimizer_agent.py](file:///c:/proyectos/Katalog-brain/agents/optimizer_agent.py): Agente Escritor que genera la propuesta optimizada de título, descripción HTML y tags de SEO.
    *   [critic_agent.py](file:///c:/proyectos/Katalog-brain/agents/critic_agent.py): Agente Juez que valida si la propuesta cumple con las reglas de marca (evitando palabras prohibidas y limitando caracteres).
    *   Otros agentes de soporte: `ingestion_agent.py`, `inspector_agent.py`, `reclassifier_agent.py`.
*   **`scripts/`**: Contiene **utilidades operativas** y scripts de administración:
    *   [visualize_graph.py](file:///c:/proyectos/Katalog-brain/scripts/visualize_graph.py): Script para imprimir y visualizar de forma gráfica e interactiva la estructura del grafo en formato ASCII directo en la terminal.

---

## 🗺️ Diccionario de Nodos (`core/graph.py`)

A continuación se detalla la lista de todos los **nodos** (pasos del flujo) que componen el grafo principal del agente, en qué línea del archivo [graph.py](file:///c:/proyectos/Katalog-brain/core/graph.py) residen y qué sub-agente o servicio ejecutan:

| Nombre del Nodo | Función en `graph.py` | Línea de Código | Agente de IA o Servicio Asociado |
| :--- | :--- | :--- | :--- |
| **`start_processing`** | `start_processing` | [L255](file:///c:/proyectos/Katalog-brain/core/graph.py#L255) | Servicio Supabase. Cambia el estado del producto a `PROCESSING`. |
| **`fetch_data`** | `fetch_db_data` | [L292](file:///c:/proyectos/Katalog-brain/core/graph.py#L292) | API de Supabase e Integración Shopify. Descarga metadatos del producto, reglas de marca y consulta la taxonomía predictiva. |
| **`memory`** | `retrieve_memory_letta` | [L403](file:///c:/proyectos/Katalog-brain/core/graph.py#L403) | API de Letta. Recupera la memoria de largo plazo e insights históricos. |
| **`retrieve_knowledge`**| `retrieve_knowledge` | [L420](file:///c:/proyectos/Katalog-brain/core/graph.py#L420) | Google Gemini Vector Embeddings y RPC. Consulta el almacén de RAG global en base a similitud semántica. |
| **`ai_writer`** | `audit_and_write_pydantic`| [L464](file:///c:/proyectos/Katalog-brain/core/graph.py#L464) | **Optimizer Agent** (`run_optimizer_with_fallback`). Redacta el nuevo título, SEO tags y descripción bajo frameworks de conversión (PAS/FAB). |
| **`critic`** | `review_proposal` | [L542](file:///c:/proyectos/Katalog-brain/core/graph.py#L542) | **Critic Agent** (`run_critic_with_fallback`). Actúa como el Juez evaluando la propuesta redactada. |
| **`save_db`** | `save_to_supabase` | [L610](file:///c:/proyectos/Katalog-brain/core/graph.py#L610) | Servicio Supabase. Persiste la propuesta aprobada en la base de datos como `READY_TO_PUBLISH`. |
| **`needs_optimization`**| `mark_needs_optimization`| [L655](file:///c:/proyectos/Katalog-brain/core/graph.py#L655) | Servicio Supabase. Registra los errores encontrados cuando el Juez rechaza la propuesta y se agotan los reintentos. |
| **`publish_to_shopify`**| `publish_to_shopify_node`| [L691](file:///c:/proyectos/Katalog-brain/core/graph.py#L691) | API Shopify Admin GraphQL. Publica el título y HTML optimizados en la tienda del cliente final. |
| **`error_handler`** | `error_handler` | [L772](file:///c:/proyectos/Katalog-brain/core/graph.py#L772) | Manejo de Errores. Actualiza los logs de auditoría ante cualquier excepción imprevista. |

---

## 🛠️ Guía de Edición Rápida (Paso a Paso)

### 1. Cómo cambiar un Prompt de la IA
Los prompts lógicos e instrucciones para el redactor y el juez viven en los agentes dentro de la carpeta `agents/`:

1.  **Localizar el agente adecuado**:
    *   Si quieres cambiar la forma en que escribe la descripción o título, abre [optimizer_agent.py](file:///c:/proyectos/Katalog-brain/agents/optimizer_agent.py).
    *   Si quieres cambiar el criterio de validación del Juez, abre [critic_agent.py](file:///c:/proyectos/Katalog-brain/agents/critic_agent.py).
2.  **Modificar el prompt del sistema**:
    *   Modifica el string multiline `system_prompt` configurado en el objeto de agente correspondiente.
3.  **Para inyecciones de contexto específicas del grafo**:
    *   Si necesitas inyectar variables en el prompt del escritor que dependen de los nodos anteriores (como RAG o Taxonomía), abre [graph.py:L464](file:///c:/proyectos/Katalog-brain/core/graph.py#L464) en la función `audit_and_write_pydantic` y edita la plantilla f-string `prompt`.

---

### 2. Cómo agregar una nueva variable al Estado (`State`)
El estado almacena los datos temporales del flujo de un producto. Si quieres usar datos nuevos en diferentes nodos:

1.  **Añadir en `state.py`**:
    Abre [state.py](file:///c:/proyectos/Katalog-brain/core/state.py) y declara la variable en la clase `KatalogState` con su tipado (ej. `NotRequired[str]` si puede ser nulo o `Required[int]` si es obligatorio):
    ```python
    class KatalogState(TypedDict, total=False):
        # ... variables existentes
        nueva_variable_contexto: NotRequired[str]
    ```
2.  **Escribir en el Nodo Emisor**:
    En cualquier nodo productor en [graph.py](file:///c:/proyectos/Katalog-brain/core/graph.py), incluye la variable en el diccionario de retorno:
    ```python
    async def mi_nodo_emisor(state: KatalogState) -> dict[str, Any]:
        return {"nueva_variable_contexto": "valor calculado"}
    ```
3.  **Leer en el Nodo Receptor**:
    En el nodo receptor en [graph.py](file:///c:/proyectos/Katalog-brain/core/graph.py), lee la variable directamente de `state`:
    ```python
    async def mi_nodo_receptor(state: KatalogState) -> dict[str, Any]:
        valor = state.get("nueva_variable_contexto", "valor_default")
        # usar valor...
    ```

---

### 3. Cómo saltarse o eliminar un nodo del Grafo
Para eliminar o re-enrutar un nodo del flujo operativo del agente de forma segura (por ejemplo, saltarse el paso de la memoria de `letta`):

1.  **Localizar la construcción del grafo**:
    Ve al final de [graph.py](file:///c:/proyectos/Katalog-brain/core/graph.py) dentro de la función `build_graph()` (Línea ~800).
2.  **Modificar las conexiones condicionales (`Edges`)**:
    *   Identifica qué nodo apunta al que deseas eliminar. Por ejemplo, `fetch_data` solía apuntar a `memory` mediante la regla condicional `route_after_fetch`.
    *   Cambia el destino en la configuración de la arista. Si queremos saltarnos `memory` e ir directo de `fetch_data` a `retrieve_knowledge`:
        ```python
        # En la definición del router condicional o en los enlaces de build_graph():
        workflow.add_conditional_edges(
            "fetch_data",
            route_after_fetch,
            {
                "retrieve_knowledge": "retrieve_knowledge", # Cambiar "memory" por el siguiente nodo
                "publish_to_shopify": "publish_to_shopify",
                "error_handler": "error_handler",
            },
        )
        ```
3.  **(Opcional) Eliminar la definición física**:
    Si ya no se usará el nodo, puedes comentar o borrar su registro en `build_graph()`:
    ```python
    # workflow.add_node("memory", retrieve_memory_letta)  # Comentado para eliminarlo del flujo de procesamiento
    ```
    Y finalmente ejecuta en tu terminal `python scripts/visualize_graph.py` para corroborar visualmente que la conexión y el nodo se han alterado exitosamente en el mapa del flujo.
