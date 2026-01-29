# Docker 安装指南

## 快速安装（推荐）

```bash
# 下载并执行官方安装脚本
curl -fsSL https://get.docker.com | sudo sh

# 启动 Docker 并设置开机自启
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装
docker --version

sudo docker run hello-world
```