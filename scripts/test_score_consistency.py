# scripts/test_score_consistency.py
import os, sys, time, statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from agents.inspector_agent import inspector_agent
from core.model_config import INSPECTOR_MODEL

USER_ID = "e06774a7-af54-4c07-bf56-6aca7b69b78f"
RUNS = 5
PAUSE = 1.5

url = os.getenv("SUPABASE_URL")
key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY")
       or os.getenv("SUPABASE_SERVICE_KEY")
       or os.getenv("SUPABASE_KEY"))
if not url or not key:
    sys.exit("Faltan SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY en el .env")

sb = create_client(url, key)

# --- 1. Traer todo y descubrir el esquema real ---
rows = (sb.table("shopify_products")
          .select("*")
          .eq("user_id", USER_ID)
          .limit(200)
          .execute().data)

if not rows:
    sys.exit(f"Cero productos para user_id {USER_ID}")

cols = list(rows[0].keys())
print("\nCOLUMNAS REALES DE shopify_products:")
print("  " + ", ".join(cols) + "\n")

def pick(candidatos, obligatorio=True):
    for c in candidatos:
        if c in cols:
            return c
    # fallback: cualquier columna que contenga la palabra clave
    for c in cols:
        for cand in candidatos:
            if cand.split("_")[0] in c.lower():
                return c
    if obligatorio:
        sys.exit(f"No encuentro ninguna de estas columnas: {candidatos}")
    return None

COL_ID    = pick(["shopify_id", "shopify_product_id", "product_id", "id"])
COL_TITLE = pick(["title", "product_title", "name", "product_name", "titulo"])
COL_DESC  = pick(["body_html", "description", "body", "product_description", "descripcion"])
COL_SCORE = pick(["audit_score", "score", "health_score"], obligatorio=False)

print(f"Usando -> id: {COL_ID} | title: {COL_TITLE} | desc: {COL_DESC} | score: {COL_SCORE}\n")

# --- 2. Elegir 3 productos con contenido real ---
validos = [r for r in rows if (r.get(COL_TITLE) or "").strip() and (r.get(COL_DESC) or "").strip()]
if len(validos) < 3:
    sys.exit(f"Solo {len(validos)} productos tienen titulo Y descripcion. Necesito 3.")

if COL_SCORE and any(r.get(COL_SCORE) is not None for r in validos):
    con_score = sorted([r for r in validos if r.get(COL_SCORE) is not None],
                       key=lambda r: r[COL_SCORE])
    targets = [con_score[0], con_score[len(con_score)//2], con_score[-1]]
else:
    print("(sin audit_score utilizable, tomo 3 productos cualesquiera)")
    targets = validos[:3]

def audit(prompt, settings):
    r = inspector_agent.run_sync(prompt, model_settings=settings)
    return r.output.score, r.output.reason

print(f"Modelo inspector: {INSPECTOR_MODEL}")
print(f"Corridas: {RUNS} por config, por producto")
print("=" * 78)

resumen = []

for p in targets:
    prompt = f"Title: {p[COL_TITLE]}\nDescription: {p[COL_DESC]}"
    print(f"\nID {p[COL_ID]} | {(p[COL_TITLE] or '')[:45]}")
    if COL_SCORE:
        print(f"  score guardado en DB: {p.get(COL_SCORE)}")

    for etiqueta, settings in [("ACTUAL (sin temperature)", None),
                               ("temperature=0", {"temperature": 0})]:
        scores, razones = [], []
        for i in range(RUNS):
            try:
                s, why = audit(prompt, settings)
                scores.append(s); razones.append(why)
            except Exception as e:
                print(f"    corrida {i+1} FALLO: {type(e).__name__}: {e}")
            time.sleep(PAUSE)

        if not scores:
            print(f"  [{etiqueta}] sin datos"); continue

        spread = max(scores) - min(scores)
        print(f"  [{etiqueta}]")
        print(f"    scores : {scores}")
        print(f"    min {min(scores)} | max {max(scores)} | spread {spread} | "
              f"media {statistics.mean(scores):.1f} | sd {statistics.pstdev(scores):.2f}")
        print(f"    razon 1: {razones[0][:70]}")
        resumen.append((p[COL_ID], etiqueta, spread))

print("\n" + "=" * 78)
print("RESUMEN DE DISPERSION")
for pid, etiqueta, spread in resumen:
    print(f"  {str(pid):>14} | {etiqueta:<26} | spread {spread:>3}")

peor_actual = max((s for _, e, s in resumen if e.startswith("ACTUAL")), default=0)
peor_temp0  = max((s for _, e, s in resumen if e == "temperature=0"), default=0)
print(f"\n  Peor spread ACTUAL       : {peor_actual}")
print(f"  Peor spread temperature=0: {peor_temp0}")
print("\n  <=3  -> sirve como gate de publicacion")
print("  4-10 -> sirve solo con margen de seguridad")
print("  >10  -> no sirve, hay que ir a criterios deterministas")
print("=" * 78 + "\n")