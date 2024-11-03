import ollama
import requests
import json

'''
To run first start ollama with `ollama serve` in a terminal window.
'''

def call_and_stream_results(query, context):
    r = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2',
            'prompt': query,
            'context': context,
        },
        stream=True)
    r.raise_for_status()
    
    full_context = []
    full_response = ""
    for l in r.iter_lines():
        response = json.loads(l)

        full_response += response.get('response', '')
        full_context.extend(response.get('context', ''))

    print("Response:\n\t" + full_response)
    return full_context


def main():
    ## This array stores every message returned by the model in full.
    context = []
    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print("\n")
        print(context)
        context = call_and_stream_results(user_input, context)
        print("\n")

if __name__ == "__main__":
    main()
