import pandas as pd

def select_samples(metadata, n_samples, seed=42):
	"""
	Select n_samples random samples from the metadata
	"""

	metadata = metadata[metadata['path'].str.contains('icl')] # filter ICL samples
	
	# Sample n_samples from each class
	sampled_by_class = {}
	for diagnosis in metadata['diagnosis'].unique():
		sampled_by_class[diagnosis] = metadata[metadata['diagnosis'] == diagnosis].sample(n=n_samples, random_state=seed)
	
	patients = pd.concat([sampled_by_class[diagnosis] for diagnosis in metadata['diagnosis'].unique()], ignore_index=True)
	
	# Reset index to have continuous numbering
	patients = patients.reset_index(drop=True)

	return patients

def get_few_shot_mappings(few_shot_samples_metadata, label_replacements):
	"""
	Create few-shot mappings, that is, a dictionary with the label as key and the list of paths as value
	Example:
	{
		'LBBB': ['path1', 'path2', 'path3'],
		'Normal': ['path4', 'path5', 'path6']
	}
	"""
	# dictionary with the description as key and the path as value

	few_shot_mappings = {
		label_replacements[str(diagnosis)]: few_shot_samples_metadata[few_shot_samples_metadata['diagnosis'] == diagnosis]['path'].tolist()
		for diagnosis in few_shot_samples_metadata['diagnosis'].unique()
	}

	return few_shot_mappings