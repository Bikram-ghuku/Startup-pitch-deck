"""
Main entry point for the market research system.
"""
import os
import asyncio
import json
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
    A mobile app that uses AI to analyze photos of skin conditions and provide 
    preliminary assessments, recommending whether a dermatologist visit is needed. 
    The app maintains a history of skin changes over time and can share reports 
    directly with healthcare providers.
    """

    try:
        # Run the market research workflow
        print("Starting market research workflow...")
        result = await market_research_graph(
            product_description=product_description,
            groq_api_key=groq_api_key,
            max_iterations=3  # Maximum innovation-debate iterations
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
        print("Final Innovation Proposal:")
        print(result["final_innovation"])

        print("\nPitch Deck Generation:")
        print("--------------------")
        # Save the pitch deck
        from tools.save_pitch_deck_ppt import save_pitch_deck_ppt
        pitch_deck_result = save_pitch_deck_ppt(
            content=result["pitch_deck"],
            filename="ai_skin_analysis_pitch.pptx"
        )
        print(pitch_deck_result)

    except Exception as e:
        print(f"Error in market research workflow: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
