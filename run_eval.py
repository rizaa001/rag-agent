from agent import agent_executor
from eval_questions import eval_set

results = []

for i, item in enumerate(eval_set, 1):
    question = item["question"]
    expected_tool = item["expected_tool"]

    print(f"\n{'='*60}")
    print(f"Q{i}: {question}")
    print(f"Expected tool: {expected_tool}")
    print('='*60)

    response = agent_executor.invoke({"input": question})
    answer = response["output"]

    # Check which tool(s) were actually called from intermediate_steps
    tools_used = []
    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            action = step[0]
            tools_used.append(action.tool)

    tool_correct = expected_tool in tools_used

    print(f"\nTool(s) used: {tools_used}")
    print(f"Tool correct: {'YES' if tool_correct else 'NO'}")
    print(f"\nAnswer:\n{answer}")

    results.append({
        "question": question,
        "expected_tool": expected_tool,
        "tools_used": tools_used,
        "tool_correct": tool_correct,
    })

# --- Final scorecard ---
print(f"\n\n{'#'*60}")
print("EVAL SUMMARY")
print('#'*60)
correct_count = sum(r["tool_correct"] for r in results)
print(f"Tool routing correct: {correct_count}/{len(results)}")
for r in results:
    status = "✅" if r["tool_correct"] else "❌"
    print(f"{status} {r['question']} → used {r['tools_used']}")