import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("Query Type Detector Started")
print("Type 'exit' to stop\n")

while True:

    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    result = classify_query(user_input)

    print("Bot:", result)

print("Query Type Detector Stopped")