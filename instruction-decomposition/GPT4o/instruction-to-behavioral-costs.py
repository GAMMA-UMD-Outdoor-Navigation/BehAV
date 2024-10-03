#Code to breakdown a given language instrcution and output behavioral costs for behavioral targets

#Inputs: Navigation instrcutions 
#Outputs: Landmarks, Navigation actions, Behaavioral Targets, Behavioral Actions, Behavioral Costs as lists

from openai import OpenAI
import numpy as np
import ast

client = OpenAI(api_key='') #ADD YOUR API KEY HERE

def get_instruction_breakdown(language_instruction):

    prompt = f"""
    
    "{language_instruction}", can you list the landmarks (e.g., a building), navigation actions (e.g., go forward), 
    general behavioral actions (e.g., stay on, avoid) and behavioral targets (e.g, pavement) in the paragraph given in quotes as four separate dictionaries.  
    
    Do not explain. Only output the four dictionaries.
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
          {"role": "user", "content": prompt}
      ])

    # Extract the response content which is the intrcution_breakdown
    instruction_breakdown_str = response.choices[0].message.content.strip()

    # print(instruction_breakdown_str)

    # Convert the string to a dictionary
    instruction_breakdown_dict = ast.literal_eval(instruction_breakdown_str)

    return instruction_breakdown_dict

def extract_lists_from_dict(dictionary):
    # Extract lists irrespective of the key names
    lists = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            lists[key] = np.array(value)
    return lists

def get_ith_key_list(dictionary, key_idx):
    # Get all keys and sort them
    keys = list(dictionary.keys())
    
    # Check if there are at least three keys
    if len(keys) >= key_idx:
        # Get the third key (index 2)
        ith_key = keys[key_idx-1]
        # Extract the list associated with the third key
        if isinstance(dictionary[ith_key], list):
            return np.array(dictionary[ith_key])
    return None

def get_similarity_scores(input_actions, reference_list):

    reference_list_length = len(reference_list)
    input_actions_length = len(input_actions)

    prompt = f"""
    I have a list of behavioral actions {reference_list} as a reference. 
    I want to predict the similarity of a list of input actions with the labels in the above reference list.
    Output should be an array of size ({input_actions_length} x {reference_list_length}) with a similarity score between 0 and 1. 
    Similarity scores for a given input action should sum up to 1 and should not have same values. 
    Each row of the array should indicate similarities for a single input action. 
    Do not explain. Only output the array without any texts.

    The input actions are {input_actions}
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
          {"role": "user", "content": prompt}
      ])

    # Extract the response content which is the similarity scores
    similarity_scores_str = response.choices[0].message.content.strip()

    # print(similarity_scores_str)

    # Convert the string to a list of lists (array)
    similarity_scores_list = eval(similarity_scores_str)

    # Convert the list of lists to a NumPy array
    similarity_scores_array = np.array(similarity_scores_list)

    return similarity_scores_array

def calculate_input_action_costs(similarity_scores, reference_costs):
    # Find the index of the highest similarity score for each input action
    most_similar_indices = np.argmax(similarity_scores, axis=1)

    # print(most_similar_indices)

    # Map the indices to the corresponding costs
    input_action_costs = [reference_costs[index] for index in most_similar_indices]

    return input_action_costs


language_instruction = 'Go forward until you see a stop sign, then turn left and go straight until you see a white building, stay on the pavements, stop for red traffic lights, stay away from grass'

# Input and reference lists
reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
reference_costs = [0, 0.5, 0.7, 1]

# Get instruction breakdown
instruction_breakdown = get_instruction_breakdown(language_instruction)

# Extract lists from the dictionary
extracted_lists = extract_lists_from_dict(instruction_breakdown)

# # Print extracted lists
# for key, array in extracted_lists.items():
#     print(f"{key}: {array}")

# Extract the list corresponding to a key
landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)
navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)
behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)
behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

# Print the extracted lists
print("Landmarks List:", landmark_list)
print("Navigation Actions List:", navigation_action_list)
print("Behavioral Actions List:", behavioral_action_list)
print("Behavioral Targets List:", behavioral_target_list)

# Get similarity scores for the behavioral actions w.r.t to a set of reference actions
similarity_scores = get_similarity_scores(behavioral_action_list, reference_list)

# Calculate behavioral action costs
input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)

# print("Similarity Scores:\n", similarity_scores)
print("Input Action Costs:\n", input_action_costs)

# Exmple Output

# Landmarks List: ['a stop sign' 'a white building']
# Navigation Actions List: ['Go forward' 'turn left' 'go straight']
# Behavioral Actions List: ['stay on' 'stop for' 'stay away from']
# Behavioral Targets List: ['pavements' 'red traffic lights' 'grass']
# Input Action Costs:
#  [0, 1, 0.5]

