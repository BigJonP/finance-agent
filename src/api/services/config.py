SYSTEM_PROMPT = """
You are a financial advisor that provides advice based on recent reddit market posts. 
Analyze the user's current portfolio and provide concise, personalized investment advice based on the reddit posts provided. 
Your advice should be fully based on that market context and be no more than 500 words.

Format your response with:
1. Portfolio Analysis
2. Key Recommendations
3. Risk Assessment
4. Next Steps"""

MODEL_NAME = "gpt-5-nano"

RETRIEVER_K = 3

RELEVANT_DOCUMENTS_TOP_K = 5

MAX_TOKENS = 6000


ERROR_ADVICE = """I apologize, but I encountered an error while analyzing your portfolio

Here are some general investment principles to consider:

1. **Diversification**: Spread your investments across different sectors and asset classes
2. **Risk Management**: Only invest what you can afford to lose
3. **Regular Review**: Periodically assess and rebalance your portfolio
4. **Long-term Perspective**: Focus on long-term growth rather than short-term fluctuations
5. **Research**: Always research investments thoroughly before making decisions

Please try again later, or consult with a qualified financial advisor for personalized guidance."""
