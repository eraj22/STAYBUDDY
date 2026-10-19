FROM node:22-alpine
WORKDIR /app
RUN apk add --no-cache postgresql-client
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY . .
EXPOSE 5000
CMD ["node","index.js"]
