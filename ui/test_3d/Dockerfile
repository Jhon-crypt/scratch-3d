# Build stage — Node 22
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

# Serve stage — nginx
FROM nginx:alpine
# Serve built app at /3d-test/ (host nginx proxies /3d-test/ here)
COPY --from=builder /app/dist /usr/share/nginx/html/3d-test
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
