# Makefile for Open Pilot Agent

COMPOSE = docker-compose
LOG_DIR = ./logs

.PHONY: up down logs build ps clean restart reload help

# 准备日志目录
prepare_logs:
	@mkdir -p $(LOG_DIR)
	@echo "✅ 日志目录准备完成: $(LOG_DIR)"

# 启动服务（后台）
up: prepare_logs
	@echo "🚀 启动服务..."
	$(COMPOSE) up -d

# 启动服务（前台）
up-fg: prepare_logs
	@echo "🚀 启动服务（前台）..."
	$(COMPOSE) up

# 停止服务
down:
	@echo "⏹️ 停止服务..."
	$(COMPOSE) down

# 重启服务
restart: down up

# 查看日志
logs:
	$(COMPOSE) logs -f

# 构建镜像
build:
	@echo "🔨 构建镜像..."
	$(COMPOSE) build

# 重新构建并启动
reload: build down up

# 查看服务状态
ps:
	$(COMPOSE) ps

# 清理
clean: down
	@echo "🧹 清理容器和镜像..."
	docker system prune -f

# 帮助
help:
	@echo "可用命令:"
	@echo "  up      - 启动服务（后台）"
	@echo "  up-fg   - 启动服务（前台）"
	@echo "  down    - 停止服务"
	@echo "  restart - 重启服务"
	@echo "  logs    - 查看日志"
	@echo "  build   - 构建镜像"
	@echo "  reload  - 重新构建并启动"
	@echo "  ps      - 查看服务状态"
	@echo "  clean   - 清理容器和镜像"
