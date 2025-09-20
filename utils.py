import pandas as pd
import json
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_text_file(file_path):
    """Load text from a file."""
    with open(file_path, mode="r") as f:
        return f.read()
    
def load_and_prepare_metadata(datafile_path):
	"""Load and prepare the metadata dataframe."""

	metadata = pd.read_csv(datafile_path + '/metadata.csv')

	# need to add datafile_path to get the complete path
	metadata['path'] = metadata.apply(
		lambda row: f"{datafile_path}/{row['path']}", 
		axis=1
	)

	return metadata

def format_message(message, setting):
	print(json.dumps(message[0], indent=2))
	if setting == "zero_shot":
		display_message_with_image(message[1]) # if zero shot
	else:
		print(json.dumps(message[1], indent=2))
		for i in range(2, len(message)-1):
			display_message_with_image(message[i])
		display_message_with_image(message[-1])

def display_message_with_image(message_json):
    # Parse the message
    message = json.loads(message_json) if isinstance(message_json, str) else message_json
    
    # Print the text content
    for content_item in message["content"]:
        if content_item["type"] == "text":
            print(content_item["text"])
        elif content_item["type"] == "image_url":
            # Extract the base64 string (remove the data:image/jpeg;base64, prefix)
            img_data = content_item["image_url"]["url"].split(",")[1]
            
            # Decode base64 string
            img_bytes = base64.b64decode(img_data)
            
            # Open image with PIL
            img = Image.open(io.BytesIO(img_bytes))
            
            # Display with matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
def is_valid_response(response, label_predictions, approach):
	"""Check if the model response has the expected format and valid content."""
	if approach == "cbm":
		return (
			response is not None
			and isinstance(response, dict)
			and all(key in response for key in ['thoughts', 'Right bundle branch block', 'ST segment elevation', 'T-wave inversion', 'answer'])
			and response['answer'] in label_predictions
		)
	else:
		return (
			response is not None
			and isinstance(response, dict)
			and all(key in response for key in ['thoughts', 'answer', 'score'])
			and response['answer'] in label_predictions
		)

def update_patient_results(metadata, patient, response, label_predictions, approach):
	"""Update metadata with model results, handling both valid and invalid responses."""
	patient_id = patient['patient_id']
	mask = metadata['patient_id'] == patient_id
	
	if is_valid_response(response, label_predictions, approach):
		if approach == "cbm":
			metadata.loc[mask, 'thoughts'] = response['thoughts']
			metadata.loc[mask, 'Right bundle branch block'] = response['Right bundle branch block']
			metadata.loc[mask, 'ST segment elevation'] = response['ST segment elevation']
			metadata.loc[mask, 'T-wave inversion'] = response['T-wave inversion']
			metadata.loc[mask, 'answer'] = response['answer']
			metadata.loc[mask, 'correct'] = patient['diagnosis'] == label_predictions[response['answer']]
		else:
			metadata.loc[mask, 'thoughts'] = response['thoughts']
			metadata.loc[mask, 'answer'] = response['answer']
			metadata.loc[mask, 'correct'] = patient['diagnosis'] == label_predictions[response['answer']]
			metadata.loc[mask, 'score'] = response['score']
	else:
		# Invalid/missing response - fill with None values
		metadata.loc[mask, ['thoughts', 'answer', 'correct', 'score']] = None

def calculate_binary_classification_metrics(metadata, label_predictions, positive_class=None):
	"""
	Calculate binary classification metrics (F1, Precision, Recall, Accuracy).
	
	Args:
		metadata: DataFrame with 'diagnosis', 'answer', and 'correct' columns
		label_predictions: Dictionary mapping model answers to actual labels
		positive_class: Which class to consider as positive (if None, auto-detect)
	
	Returns:
		dict: Dictionary containing F1, Precision, Recall, Accuracy scores
	"""
	# Filter out rows with missing predictions
	valid_predictions = metadata.dropna(subset=['answer', 'correct'])
	
	if len(valid_predictions) == 0:
		return {
			'accuracy': 0.0,
			'precision': 0.0,
			'recall': 0.0,
			'f1': 0.0,
			'total_samples': 0,
			'valid_predictions': 0
		}
	
	# Get true and predicted labels
	y_true = valid_predictions['diagnosis'].values
	y_pred = valid_predictions['answer'].map(label_predictions).values
	
	# Auto-detect positive class if not specified
	if positive_class is None:
		unique_classes = list(set(y_true) | set(y_pred))
		# Handle both numeric and string labels
		if len(unique_classes) == 2:
			# For binary classification, assume the higher/larger value is positive
			positive_class = max(unique_classes)
		else:
			# For string labels, try to find non-"normal" class
			try:
				positive_class = next((cls for cls in sorted(unique_classes) if 'normal' not in str(cls).lower()), unique_classes[0])
			except:
				# Fallback: use the first class alphabetically/numerically
				positive_class = sorted(unique_classes)[0]
	
	# Convert to binary (positive class = 1, others = 0)
	y_true_binary = (y_true == positive_class).astype(int)
	y_pred_binary = (y_pred == positive_class).astype(int)
	
	# Calculate metrics
	accuracy = accuracy_score(y_true_binary, y_pred_binary)
	precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
	recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
	f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
	
	# Calculate confusion matrix for additional insights
	# Ensure we get a 2x2 matrix even if one class is missing
	cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()
	
	return {
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1': f1,
		'tp': int(tp),
		'tn': int(tn),
		'fp': int(fp),
		'fn': int(fn),
		'total_samples': len(metadata),
		'valid_predictions': len(valid_predictions),
		'positive_class': positive_class
	}

def save_metrics(metrics, save_path, model_name, setting):
	"""Save metrics to a separate JSON file."""
	import json
	import numpy as np
	
	# Convert numpy types to native Python types for JSON serialization
	def convert_numpy_types(obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, dict):
			return {key: convert_numpy_types(value) for key, value in obj.items()}
		elif isinstance(obj, list):
			return [convert_numpy_types(item) for item in obj]
		else:
			return obj
	
	# Add experiment info to metrics
	metrics_with_info = {
		'experiment': {
			'model_name': model_name,
			'setting': setting,
		},
		'metrics': convert_numpy_types(metrics)
	}
	
	metrics_file = f"{save_path}/metrics_{model_name}.json"
	with open(metrics_file, 'w') as f:
		json.dump(metrics_with_info, f, indent=2)
	
	return metrics_file

def print_classification_results(metrics):
	"""Print a concise summary of classification metrics."""
	print(f"\nResults computed on {metrics['valid_predictions']}/{metrics['total_samples']} valid predictions")
	print(f"Acc: {metrics['accuracy']:.3f} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")
	print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")