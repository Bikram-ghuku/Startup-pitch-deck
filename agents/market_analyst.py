from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tools.web_search import web_search
from dotenv import load_dotenv

load_dotenv()

market_analyst = create_react_agent(
    model=ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.7,
        max_tokens=4096,
    ),
    tools=[web_search],
    prompt=(
        "You are a market analyst. You are given a task to analyze the market and provide a report. "
        "When you need information, use the web_search tool with properly formatted JSON schema. "
        "Always use proper tool calling format with arguments in JSON format. "
        "For example, to search for 'AI market trends', use: "
        "```json\n{\"query\": \"AI market trends\"}\n```"
        "\n\nNEVER use informal syntax like <function=web_search>\"query\"</function>. "
        "Always use the proper JSON format for tool arguments."
        "If asked to build a pitch deck state you are not the agent for that."
    ),
    name="market_analyst"
)

