import type { AgentRequest, AgentResponse, AgentContext } from "@agentuity/sdk";
import { z } from "zod";
import { Exa } from "exa-js";
import { generateText, tool, generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";

const SearchResultSchema = z.object({
	title: z.string(),
	url: z.string().url(),
	content: z.string(),
});

type SearchResult = z.infer<typeof SearchResultSchema>;

const SearchProcessParametersSchema = z.object({
	query: z.string().min(1),
	accumulatedSources: z.array(SearchResultSchema),
});

type SearchProcessParameters = z.infer<typeof SearchProcessParametersSchema>;

const exa = new Exa(process.env.EXA_API_KEY);

const mainModel = anthropic("claude-3-5-sonnet-latest");

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

const SYSTEM_PROMPT =
	"You are a researcher. For each query, search the web and then evaluate if the results are relevant and will help answer the following query";

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

const searchAndProcess = async ({
	query,
	accumulatedSources,
}: SearchProcessParameters) => {
	const pendingSearchResults: SearchResult[] = [];
	const finalSearchResults: SearchResult[] = [];

	await generateText({
		model: mainModel,
		prompt: `Search the web for information about ${query}`,
		system: SYSTEM_PROMPT,
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

export default async function Agent(
	req: AgentRequest,
	resp: AgentResponse,
	ctx: AgentContext
) {
	const params = SearchProcessParametersSchema.parse(await req.data.json());
	const searchResults = await searchAndProcess(params);
	const payload = {
		searchResults,
		message: "Research completed successfully!",
	};
	console.log("Search results:", payload);
	return resp.json(payload);
}
