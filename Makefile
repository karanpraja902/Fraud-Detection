# Makefile for Fraud Detection MLOps Project

.PHONY: help install test lint format clean build deploy serve monitor

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install Python dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  clean       - Clean up temporary files"
	@echo "  build       - Build Docker image"
	@echo "  deploy      - Deploy to Kubernetes"
	@echo "  serve       - Start local BentoML server"
	@echo "  monitor     - Run monitoring pipeline"
	@echo "  train       - Run training pipeline"
	@echo "  docker-up   - Start local Docker Compose stack"
	@echo "  docker-down - Stop local Docker Compose stack"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

# Code Quality
lint:
	flake8 src/ pipelines/ --max-line-length=88
	mypy src/ --ignore-missing-imports
	black --check --diff src/ pipelines/
	isort --check-only --diff src/ pipelines/

format:
	black src/ pipelines/
	isort src/ pipelines/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -f infra/k8s/deployment.yaml.updated

# Docker Operations
build:
	docker build -f infra/docker/Dockerfile.serve -t fraud-detection:latest .

docker-up:
	docker-compose -f infra/docker/docker-compose.yml up -d fraud-detector

docker-down:
	docker-compose -f infra/docker/docker-compose.yml down

# Local Development
serve:
	python retrain_and_serve.py

# ML Pipelines
train:
	python pipelines/training_pipeline.py

monitor:
	python pipelines/monitoring_pipeline.py

# Deployment
deploy:
	python pipelines/deployment_pipeline.py

# CI/CD Simulation
ci:
	@echo "Running CI checks..."
	make lint
	make test
	@echo "CI checks completed!"

# Development Setup
setup-dev:
	@echo "Setting up development environment..."
	make install
	pre-commit install
	@echo "Development environment ready!"

# Kubernetes Helpers
k8s-apply:
	kubectl apply -f infra/k8s/deployment.yaml

k8s-status:
	kubectl get pods -n fraud-detection
	kubectl get services -n fraud-detection
	kubectl get ingress -n fraud-detection

k8s-logs:
	kubectl logs -n fraud-detection -l app=fraud-detector --tail=100

k8s-port-forward:
	kubectl port-forward -n fraud-detection svc/fraud-detector-service 8080:80

# Database/Storage (if needed in future)
# db-migrate:
# 	alembic upgrade head

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# Security Scan
security-scan:
	pip install safety
	safety check --full-report
	trivy image --exit-code 1 --no-progress --format table fraud-detection:latest || echo "Trivy scan completed with vulnerabilities"
