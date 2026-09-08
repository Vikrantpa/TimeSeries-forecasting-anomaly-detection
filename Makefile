.PHONY: setup generate-data forecast anomaly test pipeline clean

PYTHON ?= python3

setup:
	pip install -r requirements.txt

generate-data:
	$(PYTHON) -m src.forecasting.generate_demand_data --seed 11
	$(PYTHON) -m src.anomaly_detection.generate_sensor_data --seed 5

forecast:
	$(PYTHON) -m src.forecasting.evaluate_forecasts

anomaly:
	$(PYTHON) -m src.anomaly_detection.evaluate_detection

test:
	$(PYTHON) -m pytest -q

# Runs both modules end-to-end: generate data, fit models, score, save charts.
pipeline: generate-data forecast anomaly
	@echo "Done. See outputs/forecasting/ and outputs/anomaly_detection/ for charts + summary CSVs."

clean:
	rm -f outputs/forecasting/* outputs/anomaly_detection/*
	mkdir -p outputs/forecasting outputs/anomaly_detection
