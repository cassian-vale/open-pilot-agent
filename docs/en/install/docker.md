# Docker Installation Guide

## Quick Installation (Recommended)

```bash
# Download and execute the official installation script
curl -fsSL https://get.docker.com | sudo sh

# Start Docker and enable it to start on boot
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version

sudo docker run hello-world
```
