import json



def get_json_file(path):
	
	with open(path, "r") as f:
		data=json.load(f)
	return data
    