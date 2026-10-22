#!/bin/sh
# Gera /config.js a partir de variáveis de ambiente (executado pelo entrypoint do nginx).
set -eu
cat > /usr/share/nginx/html/config.js <<EOF
window.__APP_CONFIG__ = { API_BASE_URL: "${API_BASE_URL:-/api}", ENVIRONMENT: "${APP_ENVIRONMENT:-local}" };
EOF
