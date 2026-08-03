"""
core/auth.py
Verificación de identidad para endpoints HTTP de Katalog-Brain.

Principio: ningún endpoint debe confiar en un user_id que llegue en el
body o en query params de la petición. El único user_id confiable es el
que devuelve get_current_user_id(), porque viene de un JWT verificado
contra el servidor de Auth de Supabase.
"""

import logging
import os

import httpx
from fastapi import Header, HTTPException

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

_AUTH_TIMEOUT = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=5.0)


async def _fetch_user(token: str) -> dict | None:
    """
    Verifica el token contra el endpoint /auth/v1/user de Supabase.

    Devuelve el payload del usuario si el token es válido, None si Supabase
    lo rechaza (401/403), y lanza HTTPException(503) si la red falla tras
    reintentar. Distinguir estos dos casos es deliberado: un fallo de red
    NO debe reportarse como token inválido.

    Se usa un cliente efímero por petición a propósito. Un cliente de larga
    vida acumula conexiones muertas en el pool y produce timeouts de
    handshake TLS intermitentes.
    """
    url = f"{_AUTH_SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": _AUTH_SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
    }

    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=_AUTH_TIMEOUT) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()

            if response.status_code in (401, 403):
                return None

            last_error = RuntimeError(
                f"Auth de Supabase respondió {response.status_code}"
            )
            logger.warning(
                f"[auth] intento {attempt}: respuesta inesperada "
                f"{response.status_code}"
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = e
            logger.warning(f"[auth] intento {attempt} falló por red: {e}")

    raise HTTPException(
        status_code=503,
        detail="Servicio de autenticación no disponible. Reintenta en unos segundos.",
    ) from last_error


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

    Lanza HTTPException(503) si el servicio de Auth de Supabase no está
    disponible (fallo de red tras reintentar). Un 503 intencional evita
    diagnosticar un problema de red como un token inválido.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Falta el header 'Authorization: Bearer <token>'.",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token vacío.")

    payload = await _fetch_user(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")

    user_id = payload.get("id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="No se pudo identificar al usuario a partir del token.",
        )

    return user_id
