FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app

RUN npm install -g serve

COPY --from=builder /app/dist ./dist

CMD ["sh", "-c", "if [ -z \"$PORT\" ] || [ \"$PORT\" = \"undefined\" ]; then PORT=3000; fi; serve -s dist -l $PORT"]
