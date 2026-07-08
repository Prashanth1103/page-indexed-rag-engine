import asyncio
import json
from typing import Sequence
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from rag_engine.retriever import HierarchicalPageRetriever
from rag_engine.schemas import DocumentIndex, ResearchReport
from rag_engine.telemetry import telemetry


class DeepResearcherCrew:
    """
    Multi-Agent CrewAI workflow orchestrating:
      1. Intent Decomposition Planner Agent
      2. Page-Indexed Retrieval Specialist Agent (bound to HierarchicalPageRetriever tool)
      3. Citation Verification & Synthesis Agent (enforcing Pydantic ResearchReport schema)
    """

    def __init__(self, corpus: Sequence[DocumentIndex], model_name: str = "gpt-4o") -> None:
        self.retriever = HierarchicalPageRetriever(corpus=corpus)
        self.model_name = model_name

    def build_crew(self, user_query: str) -> Crew:
        retriever_ref = self.retriever

        @tool("page_indexed_hierarchical_search")
        def page_indexed_search_tool(subquery: str) -> str:
            """Executes the 4-stage page-indexed retrieval pipeline and returns JSON-serialized page citations."""
            report = asyncio.run(retriever_ref.retrieve_and_synthesize(subquery, top_k=5))
            return report.model_dump_json(indent=2)

        planner_agent = Agent(
            role="Commercial Research Intent Planner",
            goal=(
                "Analyze user prompts and expand ambiguous or broad questions into structured, "
                "intent-driven sub-queries covering pricing assumptions, volume drivers, and margin impact."
            ),
            backstory=(
                "Senior RGM & Forecasting strategist who prevents silent recall failures by ensuring "
                "every dimension of a vague business question is explicitly queried."
            ),
            llm=self.model_name,
            verbose=False,
            allow_delegation=False,
        )

        retrieval_agent = Agent(
            role="Page-Indexed Document Retrieval Specialist",
            goal=(
                "Execute 4-stage hierarchical retrieval over page-indexed enterprise documents and "
                "table summaries, returning only high-precision sections with exact page numbers."
            ),
            backstory=(
                "Specialist in hierarchical document indexing who never relies on naive fixed-window chunking."
            ),
            tools=[page_indexed_search_tool],
            llm=self.model_name,
            verbose=False,
            allow_delegation=False,
        )

        verifier_agent = Agent(
            role="Citation Precision & Faithfulness Auditor",
            goal=(
                "Synthesize the final research report strictly from verified page sections and guarantee "
                "every statement carries an auditable (doc_id, page_number, section_id) citation."
            ),
            backstory=(
                "Enterprise compliance auditor enforcing 94%+ RAGAS citation precision with zero hallucinated claims."
            ),
            llm=self.model_name,
            verbose=False,
            allow_delegation=False,
        )

        plan_task = Task(
            description=(
                f"Decompose the following user query into intent-driven sub-queries if vague, "
                f"or keep targeted if specific: '{user_query}'"
            ),
            expected_output="A JSON list of targeted sub-queries.",
            agent=planner_agent,
        )

        retrieve_task = Task(
            description=(
                "Run `page_indexed_hierarchical_search` for each planned sub-query and consolidate "
                "all verified page sections and table summaries."
            ),
            expected_output="Consolidated JSON evidence containing doc_id, page_number, section_id, and raw_content.",
            agent=retrieval_agent,
            context=[plan_task],
        )

        synthesize_task = Task(
            description=(
                f"Produce the final verified ResearchReport answering '{user_query}' using only "
                "the retrieved page-indexed sections."
            ),
            expected_output="A validated ResearchReport JSON object with exact page citations.",
            agent=verifier_agent,
            context=[plan_task, retrieve_task],
            output_pydantic=ResearchReport,
        )

        return Crew(
            agents=[planner_agent, retrieval_agent, verifier_agent],
            tasks=[plan_task, retrieve_task, synthesize_task],
            process=Process.sequential,
            verbose=False,
        )

    @telemetry(span_name="crew.kickoff_deep_researcher", layer="crew")
    def run(self, user_query: str) -> ResearchReport:
        crew = self.build_crew(user_query)
        result = crew.kickoff()
        if hasattr(result, "pydantic") and isinstance(result.pydantic, ResearchReport):
            return result.pydantic
        raw_dict = json.loads(str(result))
        return ResearchReport.model_validate(raw_dict)
