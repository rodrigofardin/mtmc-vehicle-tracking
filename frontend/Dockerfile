# ---- build ----
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ---- runtime: nginx sem privilégios (uid 101), sem Python/PyTorch/OpenCV ----
FROM nginxinc/nginx-unprivileged:1.27-alpine
USER root
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.d/40-app-config.sh /docker-entrypoint.d/40-app-config.sh
COPY --from=build /app/dist /usr/share/nginx/html
RUN chmod +x /docker-entrypoint.d/40-app-config.sh && chown -R 101:101 /usr/share/nginx/html
USER 101

# Configuração em tempo de execução (nada de endereço de backend dentro do bundle)
ENV BACKEND_URL=http://mtmc-api:8000 \
    DNS_RESOLVER=127.0.0.11 \
    API_BASE_URL=/api \
    APP_ENVIRONMENT=local
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
