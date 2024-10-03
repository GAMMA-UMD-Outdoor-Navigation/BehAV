from openai import OpenAI
import numpy as np

client = OpenAI(api_key='') #ADD YOUR API KEY HERE

def get_similarity_scores(input_actions, reference_list):

    reference_list_length = len(reference_list)
    input_actions_length = len(input_actions)

    prompt = f"""
    I have a list of behavioral actions {reference_list} as a reference. 
    I want to predict the similarity of a list of input actions with the labels in the above reference list.
    Only output an array of size ({input_actions_length} x {reference_list_length}) with a similarity score between 0 and 1. 
    Similarity scores for a given input action should sum up to 1 and should not have same values. 
    Each row of the vector should indicate similarities for a single input action. 
    Do not explain.

    The input actions are {input_actions}
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
          {"role": "user", "content": prompt}
      ])

    # Extract the response content which is the similarity scores
    similarity_scores_str = response.choices[0].message.content.strip()

    # Convert the string to a list of lists (array)
    similarity_scores_list = eval(similarity_scores_str)

    # Convert the list of lists to a NumPy array
    similarity_scores_array = np.array(similarity_scores_list)

    return similarity_scores_array

def calculate_input_action_costs(similarity_scores, reference_costs):
    # Find the index of the highest similarity score for each input action
    most_similar_indices = np.argmax(similarity_scores, axis=1)

    print(most_similar_indices)

    # Map the indices to the corresponding costs
    input_action_costs = [reference_costs[index] for index in most_similar_indices]

    return input_action_costs


# Input and reference lists
reference_list = ['Prefer', 'Avoid', 'Yield', 'Stop']
reference_costs = [0, 0.5, 0.7, 1]

input_actions = ['follow', 'stay away']

# Get similarity scores
similarity_scores = get_similarity_scores(input_actions, reference_list)

# Calculate input action costs
input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)

print("Similarity Scores:\n", similarity_scores)
print("Input Action Costs:\n", input_action_costs)

