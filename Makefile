.PHONY: help install ingest query evaluate test api ui docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make ingest        Ingest sample documents into vector database"
	@echo "  make query         Interactive query session"
	@echo "  make evaluate      Run benchmark evaluation suite"
	@echo "  make test          Run pytest suite"
	@echo "  make api           Start FastAPI backend server"
	@echo "  make ui            Start Streamlit UI application"
	@echo "  make docker-up     Start all services with docker-compose"
	@echo "  make docker-down   Stop docker containers"
	@echo "  make clean         Clean build artifacts and caches"

install:
	pip install -r requirements.txt

ingest:
	python cli.py ingest --dir ./data/sample_docs

query:
	python cli.py interactive

evaluate:
	python cli.py benchmark --dataset ./data/evaluation/golden_qa_dataset.json

test:
	pytest -v tests/

api:
	python cli.py serve --host 0.0.0.0 --port 8000

ui:
	python cli.py ui --port 8501

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ build/ dist/
