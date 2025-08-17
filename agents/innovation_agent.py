from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tools.web_search import web_search
from dotenv import load_dotenv

load_dotenv()

innovation_strategist = create_react_agent(
    model=ChatGroq(
        model="openai/gpt-oss-120b", 
        temperature=0.8,  # Higher temperature for more innovative thinking
        max_tokens=4096,
    ),
    tools=[web_search],
    prompt=(
        "You are an innovation strategist. You are given a task to suggest unique features or differentiators for a product. "
        "When you need information, use the web_search tool with properly formatted JSON schema. "
        "Always use proper tool calling format with arguments in JSON format. "
        "For example, to search for 'innovative product features', use: "
        "```json\n{\"query\": \"innovative product features\"}\n```"
        "\n\nNEVER use informal syntax like <function=web_search>\"query\"</function>. "
        "Always use the proper JSON format for tool arguments."
        "If anything out of your specilisation is asked, you should say that you are not sure and you should not make up any information."
        "Give answer in paragraph format. Do not use specilised styling like tables or lists."
    ),
    name="innovation_strategist"
)