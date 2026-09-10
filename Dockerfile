# Image du service web de prédiction du prix des maisons.
#
# Construction en deux étages : le premier installe les dépendances et entraîne
# le modèle, le second ne reçoit que le résultat. L'image finale ne contient
# donc ni uv, ni les sources d'entraînement, ni le cache d'installation.

# --- Étage 1 : dépendances et entraînement -----------------------------------
FROM python:3.12-slim AS builder

# uv résout et installe les dépendances en quelques secondes là où pip prend des
# dizaines de secondes. On le récupère depuis son image officielle plutôt que de
# l'installer avec pip, ce qui contournerait mal le problème qu'il résout.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Bytecode précompilé à la construction : le premier appel après démarrage ne
# paie plus la compilation. Copie plutôt que lien dur, le cache uv vivant sur un
# autre système de fichiers que la couche d'image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Les dépendances d'abord, seules : tant que requirements.txt ne bouge pas,
# Docker rejoue cette couche depuis son cache au lieu de tout réinstaller.
COPY requirements.txt ./
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r requirements.txt

# Le modèle est entraîné pendant la construction. On évite ainsi de versionner
# un binaire, et l'image reste autonome : rien à monter au démarrage.
COPY houses.csv train_model.py ./
RUN /opt/venv/bin/python train_model.py

# --- Étage 2 : image finale ---------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Le service n'écrit rien et n'a aucun besoin des droits d'administration. Un
# conteneur compromis qui tourne en root reste root sur son système de fichiers.
RUN useradd --create-home --uid 1000 service

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/regression.joblib ./
COPY app ./app

USER service

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
