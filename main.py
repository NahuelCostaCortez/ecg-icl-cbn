import hydra
import utils
import data
import llm
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEBUG
DEBUG = True

@hydra.main(config_path="./config/lbbb", config_name="config", version_base=None)
def main(cfg):
    # ----------- Experiment conditions -----------
	# Data settings
	shots = cfg.data.num_shots
	setting = "zero_shot" if shots == 0 else "few_shot"
	datafile_path = cfg.data.datafile_path
	label_replacements = cfg.data.label_replacements
	label_predictions = cfg.data.label_predictions
	save_path = cfg.data.save_path

	# Model settings
	model_name = cfg.model.model_name

    # system_prompt
	system_prompt_path = cfg.user_args.system_prompt_path
	user_query_path = cfg.user_args.user_query_path
	user_query_path = user_query_path.replace(".txt", f"_{setting}.txt")
	system_prompt = utils.load_text_file(system_prompt_path)
	user_query = utils.load_text_file(user_query_path)
    # --------------------------------------------
    
	# ---------------- Load data -----------------
	metadata = utils.load_and_prepare_metadata(datafile_path)

	# ICL: few-shot samples
	few_shot_samples = None
	few_shot_samples_metadata = None

	if setting != "zero_shot":
		# Select samples for few-shot learning
		few_shot_samples_metadata = data.select_samples(metadata, shots)
		few_shot_mappings = data.get_few_shot_mappings(few_shot_samples_metadata, label_replacements)
		few_shot_samples = llm.encode_few_shot_samples(few_shot_mappings)

	# remove ICL samples from metadata, just keep the test samples
	metadata = metadata[~metadata['path'].str.contains('icl')]
	# ---------------------------------------------
		
	# ----------------- LLM client ----------------
	client = llm.initialize_client(model_name)
	# ---------------------------------------------

	# ----------------- Run model -----------------
	for _, patient in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing patients"):
		patient_id = patient['patient_id']
		
		# Prepare input for model
		query_img = llm.encode_image(patient['path'])
		message_for_sample = llm.process_messages(system_prompt, user_query, query_img, few_shot_samples)

		if DEBUG:
			utils.format_message(message_for_sample, setting)

		# Get model prediction
		try:
			response = llm.get_model_prediction(client, model_name, message_for_sample)
		except Exception as e:
			logger.error(f"Unexpected error for patient {patient_id}: {e}")
			response = None

		# Process and store results
		utils.update_patient_results(metadata, patient, response, label_predictions)
		
		if DEBUG:
			break
	# ---------------------------------------------

	# Report results
	metrics = utils.calculate_binary_classification_metrics(metadata, label_predictions)
	utils.print_classification_results(metrics)
	
	# Save results
	os.makedirs(save_path, exist_ok=True)
	metadata.to_csv(f"{save_path}/results_{model_name}.csv", index=False)
	metrics_file = utils.save_metrics(metrics, save_path, model_name, setting)
	print(f"Results saved to: {save_path}/results_{model_name}.csv")
	print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()