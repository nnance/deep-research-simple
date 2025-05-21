import type {
	AgentContext,
	AgentRequest,
	AgentResponse,
	RemoteAgent,
} from "@agentuity/sdk";
import { generateObject, generateText, tool } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";
import { SYSTEM_PROMPT } from "../../common/prompts";

const SearchResultSchema = z.object({
	title: z.string(),
	url: z.string().url(),
	content: z.string(),
});

const SearchResultsSchema = z.object({
	searchResults: z.array(SearchResultSchema),
	message: z.string(),
});

type SearchResult = z.infer<typeof SearchResultSchema>;

const REFLECTION_PROMPT = (
	prompt: string,
	queries: string,
	learnings: { followUpQuestions: string[]; learning: string }
) => `Overall research goal: ${prompt}\n\n
          Previous search queries: ${queries}\n\n
          Follow-up questions: ${learnings.followUpQuestions.join(", ")}
          `;

type Learning = {
	learning: string;
	followUpQuestions: string[];
};

type Research = {
	query: string;
	queries: string[];
	searchResults: SearchResult[];
	learnings: Learning[];
	completedQueries: string[];
};

const accumulatedResearch: Research = {
	query: "",
	queries: [],
	searchResults: [],
	learnings: [],
	completedQueries: [],
};

const mainModel = anthropic("claude-3-5-sonnet-latest");

const generateSearchQueries = async (query: string, n = 3) => {
	const {
		object: { queries },
	} = await generateObject({
		model: mainModel,
		system: SYSTEM_PROMPT,
		prompt: `Generate ${n} search queries for the following query: ${query}`,
		schema: z.object({
			queries: z.array(z.string()).min(1).max(5),
		}),
	});
	return queries;
};

const generateLearnings = async (query: string, searchResult: SearchResult) => {
	const { object } = await generateObject({
		model: mainModel,
		system: SYSTEM_PROMPT,
		prompt: `The user is researching "${query}". The following search result were deemed relevant.
      Generate a learning and a follow-up question from the following search result:
   
      <search_result>
      ${JSON.stringify(searchResult)}
      </search_result>
      `,
		schema: z.object({
			learning: z.string(),
			followUpQuestions: z.array(z.string()),
		}),
	});
	return object;
};

async function searchWeb(query: string, researcher: RemoteAgent) {
	const response = await researcher.run({
		data: {
			query,
			accumulatedSources: accumulatedResearch.searchResults,
		},
	});
	const results = await response.data.json();
	const { searchResults } = SearchResultsSchema.parse(results);
	return searchResults;
}

const deepResearch = async (
	prompt: string,
	researcher: RemoteAgent,
	depth = 2,
	breadth = 3
) => {
	if (accumulatedResearch.query.length === 0) {
		accumulatedResearch.query = prompt;
	}

	if (depth === 0) {
		return accumulatedResearch;
	}

	const queries = await generateSearchQueries(prompt, breadth);
	accumulatedResearch.queries = queries;

	for (const query of queries) {
		console.log(`Searching the web for: ${query}`);

		const searchResults = await searchWeb(query, researcher);

		accumulatedResearch.searchResults.push(...searchResults);
		for (const searchResult of searchResults) {
			console.log(`Processing search result: ${searchResult.url}`);
			const learnings = await generateLearnings(query, searchResult);
			accumulatedResearch.learnings.push(learnings);
			accumulatedResearch.completedQueries.push(query);

			const queries = accumulatedResearch.completedQueries.join(", ");
			const newQuery = REFLECTION_PROMPT(prompt, queries, learnings);
			await deepResearch(
				newQuery,
				researcher,
				depth - 1,
				Math.ceil(breadth / 2)
			);
		}
	}
	return accumulatedResearch;
};

const researchSchema = z.object({
	query: z.string().min(1),
	deepth: z.number().min(1).max(5).optional(),
	breadth: z.number().min(1).max(5).optional(),
});

export default async function Agent(
	req: AgentRequest,
	resp: AgentResponse,
	ctx: AgentContext
) {
	const request = researchSchema.parse(await req.data.json());
	const input = request.query;
	const depth = request.deepth ?? 2;
	const breadth = request.breadth ?? 3;

	const researcher = await ctx.getAgent({ name: "researcher" });
	if (!researcher) {
		return resp.text("Researcher agent not found", {
			status: 500,
			statusText: "Agent Not Found",
		});
	}

	console.log("Starting research...");
	const research = await deepResearch(input, researcher, depth, breadth);
	console.log("Research completed!");

	const author = await ctx.getAgent({ name: "author" });
	if (!author) {
		return resp.text("Author agent not found", {
			status: 500,
			statusText: "Agent Not Found",
		});
	}
	console.log("Generating report...");
	// Make a copy of research with all properties defined
	const agentResult = await author.run({ data: research });
	const report = await agentResult.data.text();
	console.log("Report generated! report.md");

	return resp.markdown(report);
}
