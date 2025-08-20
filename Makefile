preprocess:
	python src/preprocess.py
	@echo "Preprocessing completed."

train_model: preprocess
	python src/train_model.py
	@echo "Model training completed."

evaluate_model: train_model
	python src/evaluate_model.py
	@echo "Model evaluation completed."

run_all: preprocess train_model evaluate_model
	@echo "All steps completed."