from agents.user_chat import supervisor
from dotenv import load_dotenv
from pretty_print import pretty_print_messages
import os

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print a welcome banner for the application."""
    clear_screen()
    print("=" * 80)
    print("PRODUCT DEVELOPMENT ASSISTANT".center(80))
    print("=" * 80)
    print("\nThis assistant will help you develop your product idea through:")
    print("  1. Market Research")
    print("  2. Innovation Strategy")
    print("  3. Pitch Deck Creation")
    print("\nType 'exit' at any time to quit the application.")
    print("=" * 80 + "\n")

def main():
    """Main interactive function for the product development assistant."""
    load_dotenv()
    print_banner()
    
    # Initialize conversation history
    conversation_history = []
    
    # Get initial input from user
    print("Please describe your product or business idea:")
    user_input = input("> ")
    
    if user_input.lower() == 'exit':
        print("\nThank you for using the Product Development Assistant. Goodbye!")
        return
    
    # Add initial message to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Process initial input
    print("\nProcessing your request. This may take a moment...\n")
    
    while True:
        # Stream response from supervisor agent with increased recursion limit
        for chunk in supervisor.stream(
            {"messages": conversation_history}, 
            {"recursion_limit": 50}
        ):
            try:
                pretty_print_messages(chunk)
            except Exception as e:
                print(f"Error displaying message: {str(e)}")
                print(chunk)  # Fallback to raw display
        
        # Prompt for further input
        print("\n" + "-" * 80)
        print("Your response (or type 'exit' to quit):")
        user_input = input("> ")
        
        if user_input.lower() == 'exit':
            print("\nThank you for using the Product Development Assistant. Goodbye!")
            break
        
        # Add user input to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print("\nProcessing your response...\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")