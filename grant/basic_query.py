import ollama
import requests

def call_and_stream_results(query):
    r = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2',
            'prompt': prompt,
            'context': [],
        },
        stream=True)
    r.raise_for_status()
    
    full_response = ""
    for l in r.iter_lines():
        response = json.loads(l)

        full_response += response.get('response', '')

    print(full_response)


def main():
    # context = [] # the context stores a conversation history, you can use this to make the model more context aware
    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print()
        call_and_stream_results(user_input)
        print()

if __name__ == "__main__":
    main()
