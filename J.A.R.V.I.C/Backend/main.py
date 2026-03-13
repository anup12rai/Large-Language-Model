from src.inference.pipeline import classify_query

def main():
    print("=== Query Classifier LLM ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        label = classify_query(user_input)
        print(f"Bot: {label}\n")

if __name__ == "__main__":
    main()