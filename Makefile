.PHONY: install run test models dashboard docker

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

test:
	pytest -q

models:
	python scripts/bootstrap_models.py

dashboard:
	streamlit run dashboard/streamlit_app.py

docker:
	docker compose up --build
