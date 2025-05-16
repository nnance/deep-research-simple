import type { AgentRequest, AgentResponse } from "@agentuity/sdk";
import { generateObject, generateText, tool } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";
import { Exa } from "exa-js";

const exa = new Exa(process.env.EXA_API_KEY);

type SearchResult = {
	title: string;
	url: string;
	content: string;
};

const SYSTEM_PROMPT = `You are an expert researcher. Today is ${new Date().toISOString()}. Follow these instructions when responding:
  - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
  - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
  - Be highly organized.
  - Suggest solutions that I didn't think about.
  - Be proactive and anticipate my needs.
  - Treat me as an expert in all subject matter.
  - Mistakes erode my trust, so be accurate and thorough.
  - Provide detailed explanations, I'm comfortable with lots of detail.
  - Value good arguments over authorities, the source is irrelevant.
  - Consider new technologies and contrarian ideas, not just the conventional wisdom.
  - You may use high levels of speculation or prediction, just flag it for me.
  - Use Markdown formatting.`;

const EVAL_PROMPT = (
	query: string,
	pendingResult: SearchResult,
	results: SearchResult[]
) => `Evaluate whether the search results are relevant and will help answer the following query: ${query}. If the page already exists in the existing results, mark it as irrelevant.
   
	<search_results>
	${JSON.stringify(pendingResult)}
	</search_results>

	<existing_results>
	${JSON.stringify(results.map((result) => result.url))}
	</existing_results>`;

const searchWeb = async (query: string) => {
	const { results } = await exa.searchAndContents(query, {
		numResults: 1,
		livecrawl: "always",
	});
	return results.map(
		(r) =>
			({
				title: r.title,
				url: r.url,
				content: r.text,
			} as SearchResult)
	);
};

type Learning = {
	learning: string;
	followUpQuestions: string[];
};

type Research = {
	query: string | undefined;
	queries: string[];
	searchResults: SearchResult[];
	learnings: Learning[];
	completedQueries: string[];
};

const accumulatedResearch: Research = {
	query: undefined,
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

const searchAndProcess = async (
	query: string,
	accumulatedSources: SearchResult[]
) => {
	const pendingSearchResults: SearchResult[] = [];
	const finalSearchResults: SearchResult[] = [];

	await generateText({
		model: mainModel,
		prompt: `Search the web for information about ${query}`,
		system:
			"You are a researcher. For each query, search the web and then evaluate if the results are relevant and will help answer the following query",
		maxSteps: 5,
		tools: {
			searchWeb: tool({
				description: "Search the web for information about a given query",
				parameters: z.object({
					query: z.string().min(1),
				}),
				async execute({ query }) {
					const results = await searchWeb(query);
					pendingSearchResults.push(...results);
					return results;
				},
			}),
			evaluate: tool({
				description: "Evaluate the search results",
				parameters: z.object({}),
				async execute() {
					const pendingResult = pendingSearchResults.pop();
					if (pendingResult) {
						const { object: evaluation } = await generateObject({
							model: mainModel,
							prompt: EVAL_PROMPT(query, pendingResult, accumulatedSources),
							output: "enum",
							enum: ["relevant", "irrelevant"],
						});
						if (evaluation === "relevant") {
							finalSearchResults.push(pendingResult);
						}
						console.log("Found:", pendingResult.url);
						console.log("Evaluation completed:", evaluation);
						return evaluation === "irrelevant"
							? "Search results are irrelevant. Please search again with a more specific query."
							: "Search results are relevant. End research for this query.";
						// biome-ignore lint/style/noUselessElse: <explanation>
					} else {
						return "No more search results to evaluate.";
					}
				},
			}),
		},
	});
	return finalSearchResults;
};

const deepResearch = async (prompt: string, depth = 2, breadth = 3) => {
	if (!accumulatedResearch.query) {
		accumulatedResearch.query = prompt;
	}

	if (depth === 0) {
		return accumulatedResearch;
	}

	const queries = await generateSearchQueries(prompt, breadth);
	accumulatedResearch.queries = queries;

	for (const query of queries) {
		console.log(`Searching the web for: ${query}`);
		const searchResults = await searchAndProcess(
			query,
			accumulatedResearch.searchResults
		);
		accumulatedResearch.searchResults.push(...searchResults);
		for (const searchResult of searchResults) {
			console.log(`Processing search result: ${searchResult.url}`);
			const learnings = await generateLearnings(query, searchResult);
			accumulatedResearch.learnings.push(learnings);
			accumulatedResearch.completedQueries.push(query);

			const newQuery = `Overall research goal: ${prompt}
          Previous search queries: ${accumulatedResearch.completedQueries.join(
						", "
					)}
   
          Follow-up questions: ${learnings.followUpQuestions.join(", ")}
          `;
			await deepResearch(newQuery, depth - 1, Math.ceil(breadth / 2));
		}
	}
	return accumulatedResearch;
};

const generateReport = async (research: Research) => {
	const { text } = await generateText({
		model: mainModel,
		system: SYSTEM_PROMPT,
		prompt: `Generate a report based on the following research data:\n\n 
			${JSON.stringify(research, null, 2)}`,
	});
	return text;
};

const researchSchema = z.object({
	query: z.string().min(1),
	deepth: z.number().min(1).max(5).optional(),
	breadth: z.number().min(1).max(5).optional(),
});

export default async function Agent(req: AgentRequest, resp: AgentResponse) {
	const request = researchSchema.parse(await req.data.json());
	const input = request.query;
	const depth = request.deepth ?? 2;
	const breadth = request.breadth ?? 3;

	console.log("Starting research...");
	const research = await deepResearch(input, depth, breadth);
	console.log("Research completed!");
	console.log("Generating report...");
	const report = await generateReport(research);
	console.log("Report generated! report.md");

	return resp.markdown(report);
}
