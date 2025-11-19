import ollama

response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {
            'role': 'system',
            'content': 'Support give only SQL statement',
        },
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ]
)

print(response['message']['content'])
