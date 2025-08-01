import json

def remove_duplicates(json_file_path, output_file_path):
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        plants = json.load(file)
    
    # Create a dictionary to store unique entries by common_name
    unique_plants = {}
    
    for plant in plants:
        common_name = plant['common_name']
        # Only keep the first occurrence of each common_name
        if common_name not in unique_plants:
            unique_plants[common_name] = plant
    
    # Convert the dictionary values back to a list
    unique_plants_list = list(unique_plants.values())
    
    # Write the unique plants to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(unique_plants_list, file, indent=2, ensure_ascii=False)
    
    return unique_plants_list

# Example usage:
input_file = 'assets/data.json'
output_file = 'assets/data_unique.json'
unique_plants = remove_duplicates(input_file, output_file)

print(f"Found {len(unique_plants)} unique plants after removing duplicates.")