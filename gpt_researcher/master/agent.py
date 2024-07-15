import asyncio
import time

from gpt_researcher.config import Config
from gpt_researcher.context.compression import ContextCompressor
from gpt_researcher.master.functions import *
from gpt_researcher.memory import Memory
from gpt_researcher.utils.enum import ReportType


class GPTResearcher:
    """
    GPT Researcher
    """

    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        source_urls=None,
        site_constraint=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        max_content_length=None,
        parent_query: str = "",
        subtopics: list = [],
        visited_urls: set = set(),
    ):
        self.query = query
        self.agent = agent
        self.role = role
        self.report_type = report_type
        self.report_prompt = get_prompt_by_report_type(
            self.report_type
        )  # this validates the report type
        self.websocket = websocket
        self.cfg = Config(config_path)
        self.retriever = get_retriever(self.cfg.retriever)
        self.context = []
        self.source_urls = source_urls
        self.max_content_length = max_content_length
        self.site_constraint = site_constraint
        self.memory = Memory(self.cfg.embedding_provider)
        self.visited_urls = visited_urls
        self.search_queries = []

        # Only relevant for DETAILED REPORTS
        # --------------------------------------

        # Stores the main query of the detailed report
        self.parent_query = parent_query

        # Stores all the user provided subtopics
        self.subtopics = subtopics

    async def conduct_research(self):
        print(f"🔎 Running research for '{self.query}'...")

        # Generate Agent
        if not (self.agent and self.role):
            self.agent, self.role = await choose_agent(self.query, self.cfg)
        await stream_output("logs", self.agent, self.websocket)

        # If specified, the researcher will use the given urls as the context for the research.
        if self.source_urls:
            self.context = await self.get_context_by_urls(self.source_urls)
        else:
            self.context = await self.get_context_by_search(self.query)

        time.sleep(2)

    async def write_report(self, existing_headers: list = []):
        await stream_output(
            "logs",
            f"✍️ Writing summary for research task: {self.query}...",
            self.websocket,
        )

        if self.report_type == "custom_report":
            self.role = self.cfg.agent_role if self.cfg.agent_role else self.role
        elif self.report_type == "subtopic_report":
            report = await generate_report(
                query=self.query,
                context=self.context,
                agent_role_prompt=self.role,
                report_type=self.report_type,
                websocket=self.websocket,
                cfg=self.cfg,
                main_topic=self.parent_query,
                existing_headers=existing_headers,
            )
        else:
            report = await generate_report(
                query=self.query,
                context=self.context,
                agent_role_prompt=self.role,
                report_type=self.report_type,
                websocket=self.websocket,
                cfg=self.cfg,
            )

        return report

    async def get_context_by_urls(self, urls):
        new_search_urls = await self.get_new_urls(urls)
        await stream_output(
            "logs",
            f"🧠 I will conduct my research based on the following urls: {new_search_urls}...",
            self.websocket,
        )
        scraped_sites = scrape_urls(new_search_urls, self.cfg)
        return await self.get_similar_content_by_query(self.query, scraped_sites)

    async def get_context_by_search(self, query):
        context = []
        # Generate Sub-Queries including original query
        self.search_queries = await get_sub_queries(
            query, self.role, self.cfg, self.parent_query, self.report_type
        ) + [query]

        # HACK: inject site selector for Google search in here. Would be MUCH nicer if this was done in the GoogleSerper class directly as a parameter.
        if self.site_constraint:
            self.search_queries = [
                q + f" site:{self.site_constraint}" for q in self.search_queries
            ]

        await stream_output(
            "logs",
            f"🧠 I will conduct my research based on the following queries: {self.search_queries}...",
            self.websocket,
        )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[self.process_sub_query(sub_query) for sub_query in self.search_queries]
        )
        return context

    async def process_sub_query(self, sub_query: str):
        await stream_output(
            "logs", f"\n🔎 Running research for '{sub_query}'...", self.websocket
        )

        scraped_sites = await self.scrape_sites_by_query(sub_query)
        content = await self.get_similar_content_by_query(sub_query, scraped_sites)

        if content:
            await stream_output("logs", f"📃 {content}", self.websocket)
        else:
            await stream_output(
                "logs", f"🤷 No content found for '{sub_query}'...", self.websocket
            )
        return content

    async def get_new_urls(self, url_set_input):
        new_urls = []
        for url in url_set_input:
            if url not in self.visited_urls:
                await stream_output(
                    "logs", f"✅ Adding source url to research: {url}\n", self.websocket
                )

                self.visited_urls.add(url)
                new_urls.append(url)

        return new_urls

    async def scrape_sites_by_query(self, sub_query):
        retriever = self.retriever(sub_query)
        search_results = retriever.search(
            max_results=self.cfg.max_search_results_per_query
        )
        new_search_urls = await self.get_new_urls(
            [url.get("href") for url in search_results]
        )

        await stream_output(
            "logs", f"🤔 Researching for relevant information...\n", self.websocket
        )
        scraped_content_results = scrape_urls(new_search_urls, self.cfg)

        # HACK: limit max content size:
        if self.max_content_length:
            scraped_content_results = [
                content
                for content in scraped_content_results
                if len(content) <= self.max_content_length
            ]

        return scraped_content_results

    async def get_similar_content_by_query(self, query, pages):
        await stream_output(
            "logs",
            f"📝 Getting relevant content based on query: {query}...",
            self.websocket,
        )
        # Summarize Raw Data
        context_compressor = ContextCompressor(
            documents=pages, embeddings=self.memory.get_embeddings()
        )
        # Run Tasks
        return context_compressor.get_context(query, max_results=8)

    ########################################################################################

    # DETAILED REPORT

    async def write_introduction(self):
        # Construct Report Introduction from main topic research
        introduction = await get_report_introduction(
            self.query, self.context, self.role, self.cfg, self.websocket
        )

        return introduction

    async def get_subtopics(self):
        """
        This async function generates subtopics based on user input and other parameters.

        Returns:
          The `get_subtopics` function is returning the `subtopics` that are generated by the
        `construct_subtopics` function.
        """
        await stream_output("logs", f"🤔 Generating subtopics...", self.websocket)

        subtopics = await construct_subtopics(
            task=self.query,
            data=self.context,
            config=self.cfg,
            # This is a list of user provided subtopics
            subtopics=self.subtopics,
        )

        await stream_output("logs", f"📋Subtopics: {subtopics}", self.websocket)

        return subtopics
