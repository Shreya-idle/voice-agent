# Stage 1: Build the React application
FROM node:20-alpine as builder

WORKDIR /app

# Copy package configuration
COPY package.json ./

# Install dependencies (ignoring local lockfile to avoid Windows/Linux platform conflicts)
RUN npm install

# Copy source code and build the application
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy the built assets from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy a custom nginx configuration if needed, or use default
# Expose port 80 (Railway automatically detects this port to route traffic)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
