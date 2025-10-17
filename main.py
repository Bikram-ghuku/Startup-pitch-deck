"""
Main entry point for the market research system.
"""
import os
import asyncio
from dotenv import load_dotenv
from graphs.market_research_graph import market_research_graph

async def main():
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    # Example product description
    product_description = """
   A portable urban farming kit that tracks plant health, suggests watering and nutrients, 
   and integrates with local food networks. Designed for expansion, 
   users or developers can add new crops, sensors, or community-sharing features.
    """

    try:
        # Run the market research workflow
        print("Starting market research workflow...")
        result = await market_research_graph(
            product_description=product_description,
            groq_api_key=groq_api_key,
            max_iterations=2  # Maximum innovation-debate iterations
        )

        # Print the results at each stage
        print("\nMarket Analysis Results:")
        print("------------------------")
        print("Optimistic Analysis:")
        print(result["optimistic_analysis"])
        print("\nPessimistic Analysis:")
        print(result["pessimistic_analysis"])
        print("\nMarket Synthesis:")
        print(result["market_synthesis"])

        print("\nInnovation Results:")
        print("-----------------")
        print("Current Innovation Proposal:")
        print(result["current_proposal"])
        print("\nCurrent Critique:")
        print(result["current_critique"])

    except Exception as e:
        print(f"Error in market research workflow: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
