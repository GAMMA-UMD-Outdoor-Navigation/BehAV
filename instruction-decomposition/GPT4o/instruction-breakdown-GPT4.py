from openai import OpenAI
import numpy as np
import ast

client = OpenAI(api_key='') #ADD YOUR API KEY HERE

def get_instruction_breakdown(language_instruction):

    prompt = f"""
    "{language_instruction}", can you list the landmarks, navigation actions (e.g., go forward), 
    general behavioral actions (e.g., stay on, avoid) and targets (e.g, pavement) in the paragraph given in quotes as four separate dictionaries.  

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


language_instruction = 'Go forward until you see a stop sign, then turn left and go straight until you see a white building, stay on the pavements, stop for red traffic lights, stay away from grass'

# Get instruction breakdown
instruction_breakdown = get_instruction_breakdown(language_instruction)

# Extract lists from the dictionary
extracted_lists = extract_lists_from_dict(instruction_breakdown)

# Print extracted lists
for key, array in extracted_lists.items():
    print(f"{key}: {array}")


