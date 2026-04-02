# ---------- build ----------
FROM golang:1.23-alpine AS build
RUN apk add --no-cache gcc musl-dev
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app ./cmd/...

# ---------- runtime ----------
FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=build /app /usr/local/bin/app
EXPOSE 8080
ENTRYPOINT ["app"]
