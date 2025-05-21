import type {
	AgentRequest,
	AgentResponse,
	AgentContext,
	RemoteAgent,
} from "@agentuity/sdk";
import { z } from "zod";
import { Exa } from "exa-js";
import { generateText, tool } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { SearchResultSchema, type SearchResult } from "../../common/types";

const SearchProcessParametersSchema = z.object({
	query: z.string().min(1),
	accumulatedSources: z.array(SearchResultSchema),
});

type SearchProcessParameters = z.infer<typeof SearchProcessParametersSchema>;

const EvaluationResultsSchema = z.object({
	evaluation: z.enum(["relevant", "irrelevant"]),
});

const exa = new Exa(process.env.EXA_API_KEY);

const mainModel = anthropic("claude-3-5-sonnet-latest");

const searchWeb = async (query: string) => {
	const { results } = await exa.searchAndContents(query, {
		numResults: 1,
		livecrawl: "always",
	});
	return results.map((r) => ({
		title: r.title,
		url: r.url,
		content: r.text,
	}));
};

const SYSTEM_PROMPT =
	"You are a researcher. For each query, search the web and then evaluate if the results are relevant and will help answer the following query";

const searchAndProcess = async (
	{ query, accumulatedSources }: SearchProcessParameters,
	evaluator: RemoteAgent
) => {
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
					return { results };
				},
			}),
			evaluate: tool({
				description: "Evaluate the search results",
				parameters: z.object({
					results: z.array(SearchResultSchema) || SearchResultSchema,
				}),
				async execute({ results }) {
					const pendingResult = results.pop();

					if (pendingResult) {
						const response = await evaluator.run({
							data: { query, pendingResult, accumulatedSources },
						});
						const results = await response.data.json();
						const { evaluation } = EvaluationResultsSchema.parse(results);

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

	const evaluator = await ctx.getAgent({ name: "evaluator" });
	if (!evaluator) {
		return resp.text("Evaluator agent not found", {
			status: 500,
			statusText: "Agent Not Found",
		});
	}

	const searchResults = await searchAndProcess(params, evaluator);
	const payload = {
		searchResults,
		message: "Research completed successfully!",
	};
	return resp.json(payload);
}
