"""
core/auth.py
Verificación de identidad para endpoints HTTP de Katalog-Brain.

Principio: ningún endpoint debe confiar en un user_id que llegue en el
body o en query params de la petición. El único user_id confiable es el
que devuelve get_current_user_id(), porque viene de un JWT verificado
contra el servidor de Auth de Supabase.
"""

import asyncio
import logging
import os

from fastapi import Header, HTTPException
from supabase import Client, create_client

logger = logging.getLogger(__name__)

_AUTH_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_AUTH_SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not _AUTH_SUPABASE_URL or not _AUTH_SUPABASE_ANON_KEY:
    raise RuntimeError(
        "SUPABASE_URL y SUPABASE_ANON_KEY son obligatorias para core/auth.py. "
        "Usa la anon/public key del dashboard de Supabase -- NO la "
        "service_role key. Esta verificación de identidad no necesita "
        "privilegios elevados."
    )

# Cliente dedicado y separado del cliente service_role que usa el resto
# del proyecto (core/graph.py, main.py). Mínimo privilegio: este cliente
# solo verifica JWTs de usuarios, nunca lee ni escribe tablas.
_auth_client: Client = create_client(_AUTH_SUPABASE_URL, _AUTH_SUPABASE_ANON_KEY)


async def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    """
    Dependencia de FastAPI. Úsala como:
        async def mi_endpoint(user_id: str = Depends(get_current_user_id)):

    Verifica el JWT de Supabase Auth contra el servidor de Auth (no decodifica
    el token localmente, no usa ninguna llave secreta compartida) y devuelve
    el user_id real y verificado del token.

    Lanza HTTPException(401) si:
      - falta el header Authorization
      - el header no tiene el formato 'Bearer <token>'
      - el token es inválido, está mal formado, o expiró
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Falta el header 'Authorization: Bearer <token>'.",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token vacío.")

    try:
        # auth.get_user() hace una petición HTTP real al servidor de Auth
        # de Supabase. La movemos a un hilo aparte para no bloquear el
        # event loop de FastAPI mientras esperamos la respuesta de red.
        response = await asyncio.to_thread(_auth_client.auth.get_user, token)
    except Exception as e:
        logger.warning(f"[auth] Falló la verificación de token: {e}")
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")

    user = getattr(response, "user", None)
    user_id = getattr(user, "id", None) if user else None

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="No se pudo identificar al usuario a partir del token.",
        )

    return user_id
