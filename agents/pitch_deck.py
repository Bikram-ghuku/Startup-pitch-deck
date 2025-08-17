from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tools.save_pitch_deck_ppt import save_pitch_deck_ppt
from dotenv import load_dotenv

load_dotenv()

pitch_deck_builder = create_react_agent(
    model=ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.95, 
        max_tokens=4096,
    ),
    tools=[save_pitch_deck_ppt],
    prompt=(
        "Create bold pitch decks with 20+ slides (intro, problem, solutions, your solution, how it works, market, revenue, growth, team, call to action). "
        "ONLY use save_pitch_deck_ppt tool - never output JSON directly. This should be a comprehensive pitch deck that is ready for presentation."
        
        "\nJSON format: {\"slides\": [{\"title\": \"TITLE\", \"content\": [\"Point 1\", \"Point 2\"], \"notes\": \"Notes\"}]}"
        
        "\nTool usage example:"
        "\nAction: save_pitch_deck_ppt"
        "\nAction Input: {\"content\": \"{\\\"slides\\\":[{\\\"title\\\":\\\"Title\\\",\\\"content\\\":[\\\"Point\\\"]}]}\", \"filename\": \"pitch.pptx\"}"
    ),
    name="pitch_deck_builder"
)
