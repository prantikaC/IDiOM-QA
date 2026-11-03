from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from src.core.state import MultiAgentRAGState

# Import the new prompt
from src.prompts.generator_prompt import GENERATOR_SYSTEM_PROMPT

def get_generator_agent(llm: BaseChatModel):
    """
    Factory function that returns the Generator Agent.
    Synthesizes the final answer using the verified fact ledger.
    """
    
    # Use the imported prompt template
    prompt_template = ChatPromptTemplate.from_template(GENERATOR_SYSTEM_PROMPT)
    generation_chain = prompt_template | llm

    def generator_agent(state: MultiAgentRAGState):
        query              = state["original_query"]
        past_steps         = state.get("past_steps", [])
        is_searchable      = state.get("is_web_searchable", False)
        out_of_scope_tasks = state.get("out_of_scope_tasks", [])
        raw_context        = state.get("final_context_string", "")

        print(f"--- [Generator Agent] Synthesizing final answer ---")

        # Separate resolved facts from structured sentinels.
        _SENTINELS = {"[OUT_OF_SCOPE]", "[RETRIEVAL_FAILED]"}
        resolved_steps  = [(t, f) for t, f in past_steps if f not in _SENTINELS]
        failed_steps    = [(t, f) for t, f in past_steps if f == "[RETRIEVAL_FAILED]"]

        # ==========================================
        # 1. Python-Level UX Intercepts (Saves Tokens!)
        # ==========================================
        # Completely empty plan (no retrievals ran at all).
        if not past_steps:
            query_quoted = query.strip().rstrip("?").rstrip(".")
            if not is_searchable:
                fallback_msg = (
                    f"The question regarding \"{query_quoted}\" cannot be answered as the "
                    f"available documents (CSIR/IACS/JU Office Memoranda) do not contain "
                    f"information on this topic. Because this query appears to require secure "
                    f"or internal knowledge, the public web search fallback was automatically "
                    f"blocked to protect your data."
                )
            else:
                fallback_msg = (
                    f"The question regarding \"{query_quoted}\" cannot be answered as the "
                    f"available documents do not contain information on this topic, and the "
                    f"fallback web search also did not yield a relevant answer."
                )
            print(f"--- [Generator Agent] Empty plan — outputting fallback response ---")
            return {"final_answer": fallback_msg, "messages": [AIMessage(content=fallback_msg)]}

        # All sub-tasks failed (no resolved facts) — full refusal without an LLM call.
        if not resolved_steps:
            all_missing = out_of_scope_tasks + [t for t, _ in failed_steps]
            topic = all_missing[0] if len(all_missing) == 1 else query.strip().rstrip("?").rstrip(".")
            fallback_msg = f"The available documents do not contain information on {topic}."
            print(f"--- [Generator Agent] All sub-tasks failed — outputting refusal ---")
            return {"final_answer": fallback_msg, "messages": [AIMessage(content=fallback_msg)]}

        # ==========================================
        # 2. Standard Generation (LLM call)
        # ==========================================
        ledger_text = "\n".join(
            f"- Task: {task}\n  Result: {fact}" for task, fact in resolved_steps
        )
        if failed_steps:
            ledger_text += "\n" + "\n".join(
                f"- Task: {task}\n  Result: [RETRIEVAL_FAILED — could not retrieve]"
                for task, _ in failed_steps
            )

        out_of_scope_list = (
            "\n".join(f"  - {t}" for t in out_of_scope_tasks)
            if out_of_scope_tasks else "(none)"
        )

        response = generation_chain.invoke({
            "original_query":    query,
            "past_steps_ledger": ledger_text,
            "out_of_scope_list": out_of_scope_list,
            "raw_context":       raw_context or "(not available)",
        })

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }
        
    return generator_agent