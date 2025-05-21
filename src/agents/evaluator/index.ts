import type { AgentRequest, AgentResponse, AgentContext } from "@agentuity/sdk";
import { anthropic } from "@ai-sdk/anthropic";
import { generateObject } from "ai";
import { z } from "zod";
import { SearchResultSchema, type SearchResult } from "../../common/types";

const mainModel = anthropic("claude-3-5-sonnet-latest");

const EvaluationParametersSchema = z.object({
	query: z.string().min(1),
	pendingResult: SearchResultSchema,
	accumulatedSources: z.array(SearchResultSchema),
});

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

export default async function Agent(
	req: AgentRequest,
	resp: AgentResponse,
	ctx: AgentContext
) {
	const { query, pendingResult, accumulatedSources } =
		EvaluationParametersSchema.parse(await req.data.json());

	const { object } = await generateObject({
		model: mainModel,
		prompt: EVAL_PROMPT(query, pendingResult, accumulatedSources),
		output: "enum",
		enum: ["relevant", "irrelevant"],
	});

	return resp.json({ evaluation: object });
}
