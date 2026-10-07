# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-07

First tagged release. Latest changes:

- docs: add colors to mermaid diagrams (#4)
- chore: add the MIT licence
- refactor!: drop the GET /predict endpoint, keep POST only
- docs: document setup, endpoints, Docker and remote deployment
- feat: add a Streamlit form backed by the same model
- build: containerize the service with a two-stage uv image
- test: cover both verbs, input validation and price monotonicity
- feat: serve predictions over GET and POST /predict
- chore: add the provided dataset and training script
- feat: load the regression model and expose a predict helper
- build: declare runtime and development dependencies
